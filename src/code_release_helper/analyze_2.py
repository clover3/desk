import ast
import os
import shutil
from collections import defaultdict, deque
from pathlib import Path
import importlib.util


class DependencyAnalyzer(ast.NodeVisitor):
    def __init__(self, file_path, project_root):
        self.file_path = file_path
        self.project_root = Path(project_root)
        self.current_class = None
        self.current_function = None

        # What this file defines
        self.defined_functions = {}  # name -> (lineno, end_lineno)
        self.defined_classes = {}  # name -> (lineno, end_lineno)

        # What this file depends on
        self.function_calls = set()  # Direct function calls
        self.class_references = set()  # Class instantiations/references
        self.imports = {}  # module -> [imported_names]
        self.from_imports = {}  # module -> [imported_names]

        # Internal dependencies within file
        self.internal_dependencies = defaultdict(set)  # function/class -> {dependencies}

    def visit_Import(self, node):
        for alias in node.names:
            module_name = alias.name
            imported_name = alias.asname or alias.name
            if module_name not in self.imports:
                self.imports[module_name] = []
            self.imports[module_name].append(imported_name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module
            if module_name not in self.from_imports:
                self.from_imports[module_name] = []
            for alias in node.names:
                imported_name = alias.asname or alias.name
                self.from_imports[module_name].append(imported_name)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        previous_class = self.current_class
        self.current_class = node.name
        self.defined_classes[node.name] = (node.lineno, node.end_lineno)

        # Track base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.class_references.add(base.id)
                self.internal_dependencies[node.name].add(base.id)

        self.generic_visit(node)
        self.current_class = previous_class

    def visit_FunctionDef(self, node):
        previous_function = self.current_function

        if self.current_class:
            # Method inside a class
            method_key = f"{self.current_class}.{node.name}"
            self.current_function = method_key
        else:
            # Global function
            self.current_function = node.name
            self.defined_functions[node.name] = (node.lineno, node.end_lineno)

        self.generic_visit(node)
        self.current_function = previous_function

    def visit_Call(self, node):
        # Track function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            self.function_calls.add(func_name)
            if self.current_function:
                self.internal_dependencies[self.current_function].add(func_name)
        elif isinstance(node.func, ast.Attribute):
            # Handle obj.method() calls
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr
                self.class_references.add(obj_name)
                if self.current_function:
                    self.internal_dependencies[self.current_function].add(obj_name)

        self.generic_visit(node)

    def visit_Name(self, node):
        # Track name references (potential class references)
        if isinstance(node.ctx, ast.Load):  # Only loading, not assignment
            self.class_references.add(node.id)
            if self.current_function:
                self.internal_dependencies[self.current_function].add(node.id)
        self.generic_visit(node)


class ProjectAnalyzer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.file_analyzers = {}
        self.all_python_files = []

    def discover_python_files(self):
        """Find all Python files in the project."""
        self.all_python_files = list(self.project_root.rglob("*.py"))
        return self.all_python_files

    def analyze_all_files(self):
        """Analyze all Python files in the project."""
        for file_path in self.all_python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=str(file_path))
                    analyzer = DependencyAnalyzer(file_path, self.project_root)
                    analyzer.visit(tree)
                    self.file_analyzers[file_path] = analyzer
            except (SyntaxError, UnicodeDecodeError) as e:
                print(f"Warning: Could not parse {file_path}: {e}")

    def resolve_import_to_file(self, module_name, importing_file):
        """Resolve a module import to its actual file path."""
        # Handle relative imports
        if module_name.startswith('.'):
            # Relative import - resolve based on importing file location
            importing_dir = importing_file.parent
            parts = module_name.split('.')
            relative_levels = len([p for p in parts if p == ''])

            # Go up directory levels
            target_dir = importing_dir
            for _ in range(relative_levels - 1):
                target_dir = target_dir.parent

            # Add module path
            module_parts = [p for p in parts if p != '']
            for part in module_parts:
                target_dir = target_dir / part

            # Check for __init__.py or .py file
            if (target_dir / "__init__.py").exists():
                return target_dir / "__init__.py"
            elif (target_dir.parent / f"{target_dir.name}.py").exists():
                return target_dir.parent / f"{target_dir.name}.py"
        else:
            # Absolute import within project
            module_path = self.project_root
            for part in module_name.split('.'):
                module_path = module_path / part

            # Check for package or module
            if (module_path / "__init__.py").exists():
                return module_path / "__init__.py"
            elif (module_path.parent / f"{module_path.name}.py").exists():
                return module_path.parent / f"{module_path.name}.py"

        return None

    def find_reachable_code(self, entry_files):
        """Find all code reachable from entry points using BFS."""
        entry_paths = [Path(f) for f in entry_files]

        # Start with entry files
        reachable_files = set(entry_paths)
        reachable_functions = set()
        reachable_classes = set()

        # Track one dependency reference for each file/function/class
        file_dependencies = {}  # file_path -> (referencing_file, reason)
        function_dependencies = {}  # (file, func) -> (referencing_file, referencing_func, reason)
        class_dependencies = {}  # (file, class) -> (referencing_file, referencing_func/class, reason)

        # Queue for BFS: (file_path, function_name, class_name, source_file, source_item, reason)
        queue = deque()

        # Add all functions/classes from entry files to queue
        for entry_file in entry_paths:
            if entry_file in self.file_analyzers:
                analyzer = self.file_analyzers[entry_file]

                # Mark entry files with special reason
                file_dependencies[entry_file] = (None, "ENTRY_POINT")

                # Add all top-level functions and classes as starting points
                for func_name in analyzer.defined_functions:
                    queue.append((entry_file, func_name, None, entry_file, None, "entry_point"))
                    reachable_functions.add((entry_file, func_name))
                    function_dependencies[(entry_file, func_name)] = (entry_file, None, "entry_point")

                for class_name in analyzer.defined_classes:
                    queue.append((entry_file, None, class_name, entry_file, None, "entry_point"))
                    reachable_classes.add((entry_file, class_name))
                    class_dependencies[(entry_file, class_name)] = (entry_file, None, "entry_point")

        # BFS to find all reachable code
        processed = set()

        while queue:
            current_file, func_name, class_name, source_file, source_item, reason = queue.popleft()

            # Avoid processing the same item multiple times
            item_key = (current_file, func_name, class_name)
            if item_key in processed:
                continue
            processed.add(item_key)

            if current_file not in self.file_analyzers:
                continue

            analyzer = self.file_analyzers[current_file]

            # Get dependencies for current function/class
            current_name = func_name or class_name
            if current_name in analyzer.internal_dependencies:
                for dep_name in analyzer.internal_dependencies[current_name]:
                    # Check if dependency is in same file
                    if dep_name in analyzer.defined_functions:
                        dep_key = (current_file, dep_name)
                        if dep_key not in reachable_functions:
                            reachable_functions.add(dep_key)
                            function_dependencies[dep_key] = (current_file, current_name, f"called_by_{current_name}")
                            queue.append(
                                (current_file, dep_name, None, current_file, current_name, f"called_by_{current_name}"))

                    elif dep_name in analyzer.defined_classes:
                        dep_key = (current_file, dep_name)
                        if dep_key not in reachable_classes:
                            reachable_classes.add(dep_key)
                            class_dependencies[dep_key] = (current_file, current_name, f"used_by_{current_name}")
                            queue.append(
                                (current_file, None, dep_name, current_file, current_name, f"used_by_{current_name}"))

            # Handle imports - find external dependencies
            for module_name, imported_names in analyzer.from_imports.items():
                target_file = self.resolve_import_to_file(module_name, current_file)
                if target_file and target_file in self.file_analyzers:
                    # Track file dependency if not already tracked
                    if target_file not in file_dependencies:
                        reachable_files.add(target_file)
                        file_dependencies[target_file] = (current_file, f"imported_by_{current_file.name}")

                    target_analyzer = self.file_analyzers[target_file]

                    for imported_name in imported_names:
                        if imported_name in target_analyzer.defined_functions:
                            dep_key = (target_file, imported_name)
                            if dep_key not in reachable_functions:
                                reachable_functions.add(dep_key)
                                function_dependencies[dep_key] = (
                                current_file, current_name, f"imported_from_{module_name}")
                                queue.append((target_file, imported_name, None, current_file, current_name,
                                              f"imported_from_{module_name}"))

                        elif imported_name in target_analyzer.defined_classes:
                            dep_key = (target_file, imported_name)
                            if dep_key not in reachable_classes:
                                reachable_classes.add(dep_key)
                                class_dependencies[dep_key] = (
                                current_file, current_name, f"imported_from_{module_name}")
                                queue.append((target_file, None, imported_name, current_file, current_name,
                                              f"imported_from_{module_name}"))

        return reachable_files, reachable_functions, reachable_classes, file_dependencies, function_dependencies, class_dependencies

    def remove_dead_code(self, entry_files, dry_run=True):
        """Remove all code not reachable from entry points."""
        reachable_files, reachable_functions, reachable_classes, file_deps, func_deps, class_deps = self.find_reachable_code(
            entry_files)

        print(f"Found {len(reachable_files)} reachable files")
        print(f"Found {len(reachable_functions)} reachable functions")
        print(f"Found {len(reachable_classes)} reachable classes")

        if dry_run:
            print("\n--- DRY RUN MODE ---")
            print("Files that would be kept (with one dependency reference):")
            for f in sorted(reachable_files):
                ref_file, reason = file_deps.get(f, (None, "unknown"))
                if ref_file:
                    print(f"  {f} <- referenced by: {ref_file} ({reason})")
                else:
                    print(f"  {f} <- {reason}")

            print(f"\nFunctions that would be kept (showing sample dependencies):")
            for (f, func) in sorted(reachable_functions):
                ref_file, ref_item, reason = func_deps.get((f, func), (None, None, "unknown"))
                if ref_file and ref_item:
                    print(f"  {f}::{func} <- called by: {ref_file}::{ref_item} ({reason})")
                elif ref_file:
                    print(f"  {f}::{func} <- from: {ref_file} ({reason})")
                else:
                    print(f"  {f}::{func} <- {reason}")

            print(f"\nClasses that would be kept (showing sample dependencies):")
            for (f, cls) in sorted(reachable_classes):
                ref_file, ref_item, reason = class_deps.get((f, cls), (None, None, "unknown"))
                if ref_file and ref_item:
                    print(f"  {f}::{cls} <- used by: {ref_file}::{ref_item} ({reason})")
                elif ref_file:
                    print(f"  {f}::{cls} <- from: {ref_file} ({reason})")
                else:
                    print(f"  {f}::{cls} <- {reason}")

            print(f"\nFiles that would be REMOVED ({len(self.all_python_files) - len(reachable_files)}):")
            for f in sorted(set(self.all_python_files) - reachable_files):
                print(f"  {f}")
            return

        # Actually remove files and dead code
        files_to_remove = set(self.all_python_files) - reachable_files

        # Remove entire files
        for file_path in files_to_remove:
            print(f"Removing file: {file_path}")
            backup_path = str(file_path) + ".removed"
            shutil.move(str(file_path), backup_path)

        # Remove dead functions/classes from remaining files
        for file_path in reachable_files:
            self._remove_dead_code_from_file(file_path, reachable_functions, reachable_classes)

    def _remove_dead_code_from_file(self, file_path, reachable_functions, reachable_classes):
        """Remove dead functions and classes from a specific file."""
        if file_path not in self.file_analyzers:
            return

        analyzer = self.file_analyzers[file_path]
        lines_to_remove = set()

        # Mark dead functions for removal
        for func_name, (start_line, end_line) in analyzer.defined_functions.items():
            if (file_path, func_name) not in reachable_functions:
                print(f"  Removing function '{func_name}' from {file_path}")
                for line_num in range(start_line, end_line + 1):
                    lines_to_remove.add(line_num)

        # Mark dead classes for removal
        for class_name, (start_line, end_line) in analyzer.defined_classes.items():
            if (file_path, class_name) not in reachable_classes:
                print(f"  Removing class '{class_name}' from {file_path}")
                for line_num in range(start_line, end_line + 1):
                    lines_to_remove.add(line_num)

        # If there's nothing to remove, skip file modification
        if not lines_to_remove:
            return

        # Read and rewrite the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Create backup
        shutil.copy(str(file_path), str(file_path) + ".bak")

        # Write modified content
        with open(file_path, 'w', encoding='utf-8') as f:
            for line_num, line in enumerate(lines, 1):
                if line_num not in lines_to_remove:
                    f.write(line)


def main():
    # Example usage
    project_root = r"C:\work\code\desk\src"
    entry_files = [
        r"C:\work\code\desk\src\rule_gen\reddit\dataset_build2\build_dataset2.py",
        r"C:\work\code\desk\src\rule_gen\reddit\base_bert\train2.py",
        r"C:\work\code\desk\src\rule_gen\reddit\bert_pat\train_pat.py",
        r"C:\work\code\desk\src\rule_gen\reddit\keyword_building\run6\pat_inf_filter.py",
        r"C:\work\code\desk\src\rule_gen\reddit\keyword_building\run6\score_analysis\run_kmeans.py",
    ]
    # Validate entry files exist
    for entry_file in entry_files:
        if not Path(entry_file).exists():
            print(f"Error: Entry file {entry_file} does not exist!")
            return

    analyzer = ProjectAnalyzer(project_root)

    print("Discovering Python files...")
    analyzer.discover_python_files()
    print(f"Found {len(analyzer.all_python_files)} Python files")

    print("Analyzing dependencies...")
    analyzer.analyze_all_files()

    print("Finding reachable code...")

    # First run in dry-run mode
    analyzer.remove_dead_code(entry_files, dry_run=True)

    # Ask for confirmation
    # response = input("\nProceed with actual removal? (yes/no): ").strip().lower()
    # if response == 'yes':
    #     analyzer.remove_dead_code(entry_files, dry_run=False)
    #     print("Dead code removal completed!")
    # else:
    #     print("Operation cancelled.")


if __name__ == "__main__":
    main()
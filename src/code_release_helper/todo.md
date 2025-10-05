# CriteriaMatrix Release Checklist

## Phase 1: Code Preparation
- [X] Add `hf_dataset_loader.py` to `src/rule_gen/`
- [X] Update `src/rule_gen/reddit/path_helper.py` with HF support
- [X] Add `upload_to_hf.py` to project root
- [X] Update `src/rule_gen/reddit/base_bert/train2.py` with HF initialization
- [X] Add `huggingface-hub` to requirements.txt
- [ ] Test HF loader with local files (ensure backward compatibility)

## Phase 2: Documentation Updates
- [ ] Update README.md structure:
  - [X] Add Quick Links section at top
  - [X] Move Artifacts Released section up
  - [X] Add Setup section with PYTHONPATH instructions
  - [ ] Add usage examples for exploring data
  - [X] Update all file paths to use output/ directory
  - [X] Add HuggingFace dataset link
  - [X] Document installation instructions with HF support
- [ ] Add example scripts in `examples/` directory


### 4.6 Create Dataset Card
- [X] Generate README on HuggingFace:
  ```bash
  python upload_to_hf.py create_readme
  ```
- [ ] Verify README renders correctly
- [ ] Add dataset preview (if applicable)

## Phase 5: Testing & Validation

### 5.1 Fresh Environment Test
- [ ] Clone repository in new directory
- [ ] Create fresh Python environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  pip install -r requirements.txt
  ```
- [ ] Set PYTHONPATH: `export PYTHONPATH=src`
- [ ] Test HF initialization:
  ```python
  from rule_gen.hf_dataset_loader import init_hf_manager
  init_hf_manager('youngwoo-umass/CriteriaMatrix')
  ```

### 5.2 End-to-End Tests
- [ ] Test data download:
  ```bash
  python src/rule_gen/reddit/base_bert/train2.py --sb=askscience_head
  ```
- [ ] Verify files downloaded to correct locations
- [ ] Verify cached files are reused (no re-download)
- [ ] Test with different subreddits
- [ ] Test clustering analysis (Step 4) with downloaded artifacts
- [ ] Run inference with downloaded models
- [ ] Verify all documentation paths are correct

### 5.3 Cross-Platform Test
- [ ] Test on Linux
- [ ] Test on macOS
- [ ] Test on Windows (if applicable)

## Phase 6: Repository Release

### 6.1 GitHub Release
- [ ] Create release branch
- [ ] Create GitHub release tag (e.g., `v1.0.0`)
- [ ] Write release notes including:
  - [ ] Summary of changes
  - [ ] HuggingFace dataset link
  - [ ] Installation instructions
  - [ ] Breaking changes (if any)
  - [ ] Known issues

### 6.2 Repository Metadata
- [ ] Update repository description
- [ ] Add GitHub topics: nlp, content-moderation, reddit, interpretability, pytorch, transformers
- [ ] Add repository link to HuggingFace dataset card
- [ ] Add HuggingFace link to GitHub About section

### 6.3 Transfer/Mirror to Public Repository
- [ ] Release to public repository (if currently private)
- [ ] Update all URLs in documentation
- [ ] Verify all links work after transfer

## Phase 7: External Integration

### 7.1 Academic Links
- [ ] Link to UVA Hartvigsen Lab website (if applicable)
- [ ] Add to lab's publications/resources page
- [ ] Update any related papers with dataset link

### 7.2 Community Submission
- [ ] Submit to Papers with Code
  - [ ] Add dataset
  - [ ] Link to paper (if applicable)
  - [ ] Add benchmark results
- [ ] Submit to Hugging Face Datasets trending
- [ ] Consider submission to relevant dataset catalogs

### 7.3 Announcements
- [ ] Tweet/post on social media
- [ ] Announce on relevant subreddits (r/MachineLearning, etc.)
- [ ] Post on relevant Slack/Discord communities
- [ ] Email relevant research groups
- [ ] Post on lab/university news

## Phase 8: Post-Release Maintenance

### 8.1 Monitoring
- [ ] Set up GitHub notifications for issues
- [ ] Monitor HuggingFace dataset downloads/stars
- [ ] Track Papers with Code metrics
- [ ] Monitor social media mentions

### 8.2 Support Plan
- [ ] Document common issues in FAQ
- [ ] Respond to GitHub issues within 48 hours
- [ ] Update documentation based on user feedback
- [ ] Create troubleshooting guide based on common questions

### 8.3 Versioning Strategy
- [ ] Document dataset versioning approach
- [ ] Plan for data updates (if needed)
- [ ] Establish deprecation policy for old versions
- [ ] Document how to report data issues

## Rollback Plan

If critical issues are discovered:
- [ ] Document how to revert to previous version
- [ ] Keep local backups of all uploaded data
- [ ] Test graceful d
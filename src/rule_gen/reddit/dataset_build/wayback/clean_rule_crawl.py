import os

from bs4 import BeautifulSoup

from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import load_subreddit_list


def get_reddit_archive_save_path(sb):
    rule_save_path = os.path.join(
        output_root_path, "reddit", "wayback", f"{sb}.html")
    return rule_save_path


def clean_html(file_path, output_path=None, remove_images=True):
    """
    Reads an HTML file, removes scripts, CSS, and optionally images,
    and saves the cleaned version to a new file.

    Args:
        file_path (str): Path to the input HTML file
        output_path (str, optional): Path to save the cleaned HTML. If None, returns the cleaned HTML as string
        remove_images (bool): Whether to remove image tags (default: True)

    Returns:
        str: Cleaned HTML content if output_path is None
    """
    try:
        # Read the HTML file
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # Create BeautifulSoup object
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove all script tags
        for script in soup.find_all('script'):
            script.decompose()

        # Remove all style tags
        for style in soup.find_all('style'):
            style.decompose()

        # Remove all link tags with rel="stylesheet"
        for css_link in soup.find_all('link', rel='stylesheet'):
            css_link.decompose()

        # Remove inline styles
        for tag in soup.find_all(style=True):
            del tag['style']

        # Remove images if specified
        if remove_images:
            # Remove <img> tags
            for img in soup.find_all('img'):
                img.decompose()

            # Remove background images in inline styles
            for tag in soup.find_all():
                if tag.get('background'):
                    del tag['background']
                if tag.get('style') and 'background-image' in tag['style'].lower():
                    style = tag['style'].split(';')
                    style = [s for s in style if 'background-image' not in s.lower()]
                    tag['style'] = ';'.join(style)

            # Remove picture elements
            for picture in soup.find_all('picture'):
                picture.decompose()

            # Remove svg elements
            for svg in soup.find_all('svg'):
                svg.decompose()

            # Remove figure elements that contained images
            for figure in soup.find_all('figure'):
                if not figure.find_all(string=True, recursive=False):
                    figure.decompose()

        # Clean up the HTML
        cleaned_html = soup.prettify()

        if output_path:
            # Save to new file if output path is provided
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_html)
            return f"Cleaned HTML saved to {output_path}"
        else:
            return cleaned_html

    except FileNotFoundError:
        return "Error: Input file not found"
    except Exception as e:
        return f"Error occurred: {str(e)}"


def main():
    sb_names = load_subreddit_list()
    for sb in sb_names:
        try:
            save_path = get_reddit_archive_save_path(sb)
            print(sb)
            new_rule_save_path = os.path.join(
                output_root_path, "reddit", "wayback_clean", f"{sb}.html")
            clean_html(save_path, new_rule_save_path)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()

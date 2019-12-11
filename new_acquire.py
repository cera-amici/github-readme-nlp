"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List
from bs4 import BeautifulSoup
from requests import get

import requests
import time

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Replace YOUR_GITHUB_USERNAME below with your github username.
# TODO: Add more repositories to the `repos` list.

LANGUAGES = [
'Python',
'JavaScript',
'PHP',
'Shell'
]

repos_per_language = 100
pages_per_language = 15

def get_url_middle(language, page):
    return f'&l={language}&p={page}'

def get_url_repo_list(language, page):
    url_start = 'https://github.com/search?l='
    url_end = '&q=stars%3A%3E0&s=stars&type=Repositories'
    return url_start + get_url_middle(language, page) + url_end

def get_repo_urls(list_url):
    response = get(list_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    repos = soup.select('.repo-list-item')
    urls = []
    for repo in repos:
        repo_name = repo.select('.v-align-middle')[0].get_text()
        print(repo_name)
        urls.append(repo_name)
    return urls

def get_list_urls(languages=LANGUAGES, n=pages_per_language):
    urls = []
    for language in LANGUAGES:
        for page in range(1, n+1):
            urls.append(get_url_repo_list(language, page))
    return urls

list_urls = get_list_urls()
repos = []
for i in range(4):
    start = pages_per_language * i
    end = start + pages_per_language
    repo_count = 0
    for url in list_urls[start:end]:
        if repo_count >= repos_per_language:
            break
        repos_to_add = get_repo_urls(url)
        while repos_to_add == []:
            time.sleep(10)
            repos_to_add = get_repo_urls(url)
        repos.extend(repos_to_add)
        repo_count += len(set(repos_to_add))

headers = {
    "Authorization": f"token {github_token}",
    "User-Agent": f"{github_username}"
}

if (
    headers["Authorization"] == "token "
    or headers["User-Agent"] == "YOUR_GITHUB_USERNAME"
):
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> requests.Response:
    return requests.get(url, headers=headers)


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    return github_api_request(url).json()["language"]


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    return github_api_request(url).json()


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists
    the files in a repo and returns the url that can be
    used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns
    a dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    print(repo)
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": requests.get(get_readme_download_url(contents)).text,
    }


def scrape_github_data():
    """
    Loop through all of the repos and process them. Saves the data in
    `data.json`.
    """
    data = [process_repo(repo) for repo in repos]
    json.dump(data, open("data.json", "w"))


if __name__ == "__main__":
    scrape_github_data()
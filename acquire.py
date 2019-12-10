import pandas as pd
from requests import get
from bs4 import BeautifulSoup
import os
import time

LANGUAGES = [
'Python',
'JavaScript',
'PHP',
'Shell'
]

HEADERS = {'user-agent': 'cera-amici'}

def get_url_middle(language, page):
    return f'&l={language}&p={page}'

def get_url_repo_list(language, page):
    url_start = 'https://github.com/search?l='
    url_end = '&q=stars%3A%3E0&s=stars&type=Repositories'
    return url_start + get_url_middle(language, page) + url_end

def get_readme_from_repo(url):
    response = get(url, headers=HEADERS).content
    soup = BeautifulSoup(response, 'html.parser')
    title = soup.title.get_text()
    readme_list = soup.select('.markdown-body')
    text_list = []
    for line in readme_list:
        text_list.append(line.get_text())
    text = ' '.join(text_list)
    return title, text

def get_repo_urls(list_url):
    github_base = 'https://github.com/'
    response = get(list_url, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')
    repos = soup.select('.repo-list-item')
    urls = []
    for repo in repos:
        url_end = repo.select('.v-align-middle')[0].get_text()
        urls.append(github_base + url_end)
    return urls

def get_list_urls(languages=LANGUAGES, n=10):
    urls = {}
    for language in languages:
        urls[language] = []
        for page in range(1, n+1):
            urls[language].append(get_url_repo_list(language, page))
    return urls

def get_all_repo_data(pages_per_language=2):
    dict_urls = get_list_urls(n=pages_per_language)
    urls_to_scrape = {}
    for language in dict_urls:
        language_urls = dict_urls[language]
        urls_to_scrape[language] = []
        for url in language_urls:
            urls_to_scrape[language].extend(get_repo_urls(url))
    to_df = []
    for language in urls_to_scrape:
        repo_list = urls_to_scrape[language]
        for repo in repo_list:
            repo_data = get_readme_from_repo(repo)
            while repo_data[1] is None:
                time.sleep(10)
                repo_data = get_readme_from_repo(repo)
            to_df.append({
                'title': repo_data[0],
                'readme': repo_data[1],
                'language': language
            })
    df = pd.DataFrame(to_df)
    df.to_csv('github_readmes.csv')
    return df

def get_readme_data(fresh=False):
    if os.path.exists('github_readmes.csv') and not fresh:
        df = pd.read_csv('github_readmes.csv', index_col=0)
    else:
        df = get_all_repo_data()
    return df
import argparse
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, ParseResult, urljoin
import urllib.robotparser
import urllib.request
import time
import validators
import os
import pickle
import logging
import sys
import re
import gensim.downloader as api

from es_utilities import ES
from urlDetails import URL
from utils import get_domain, normalize, save_checkpoint, load_checkpoint

model = api.load("glove-wiki-gigaword-100")

KEYWORDS = {'catholic', 'church', 'christianity', 'pope', 'popes', 'mary', 'jesus', 'bible',
            'christian', 'catholicism', 'roman', 'basilica', 'christians', 'catholics'}


def read_robotfile(link, domain):
    isAllowed = True
    USER_AGENT = 'MyCrawler(sharma.aru@husky.neu.edu)'
    path = domain + '/robots.txt'
    logging.info("reading robots.txt {}".format(path))
    delay = 1
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(path)

    try:
        rp.read()
    except TimeoutError as e:
        logging.info("Caught timeout error of {}".format(link))
        isAllowed, delay = None, None
    except requests.exceptions.RequestException as e:
        logging.info("Error handled while reading robots.txt of {}".format(link))
        isAllowed, delay = None, None
    else:
        isAllowed = rp.can_fetch(USER_AGENT, link)

        delay = rp.crawl_delay(USER_AGENT)
        if isAllowed is not None:
            if delay is None:
                delay = 0
    finally:
        logging.info("isAllowed, delay {},{}".format(isAllowed, delay))
        return isAllowed, delay



def write_to_pickle(path, id, link, title, httpheader, content, rawhtml):
    logging.info("writing the details to pickle for: {}".format(id))
    fname = os.path.join(path, "web" + str(id))
    res = {}
    res['docno'] = link
    res['head'] = str(title)
    res['httpheader'] = httpheader
    res['text'] = content
    res['rawhtml'] = rawhtml
    pickle.dump(res, open(fname, "wb"))


def scrape_page(link, delay):
    """This method crawls the page with politeness and extracts the required html content.
    Returns an id, the URL, the HTTP headers, the page contents cleaned (with term positions), the raw html, and a list of
    all in-links (known) and out-links for the page.
    200:OK, 404:not found, 403:forbidden, 301:moved permanently, 500:Internal Server Error, 304:Not modified, 401:unauthorized"""

    logging.info("Scraping the page : {}".format(link))
    try:
        response = requests.get(link)

        if response.status_code == 200:
            time.sleep(delay)
            soup = BeautifulSoup(response.content, 'html.parser')
            outlinks = []
            httpheader = response.headers
            titlePresent = soup.find('title')
            header = httpheader['Content-Type'].split(';')[0]
            if header == 'text/html':
                rawhtml = response.text

                for script in soup(["script", "style"]):
                    script.extract()

                content = soup.get_text()
                lines = (line.strip() for line in content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)

                if titlePresent is not None:
                    title = soup.find('title').text
                else:
                    title = 'NA'

                if soup.find_all("a") is not None:
                    for links in soup.find_all("a"):
                        url = links.get('href')
                        cleaned_url = clean_url(url, link)
                        outlinks.append(cleaned_url)
                    return link, title, httpheader, text, rawhtml, outlinks
            else:
                return None

    except requests.exceptions.RequestException as e:
        logging.info("Error handled while scraping {}".format(link))
        return None


def clean_url(url, host_url=""):
    """cleans the url and brings it to standard form
    res = ParseResult(scheme=u.scheme, netloc=u.hostname, path=u.path, params=u.params, query=urlencode(params), fragment=u.fragment)
    scheme, netloc, path, params, query,fragment, username, password, hostname, port
    """
    o = urlparse(url)
    if o.hostname:
        o = ParseResult(scheme=o.scheme.lower(), netloc=o.hostname.lower(), path=o.path.replace('//','/'),
                        params=o.params, query=o.query, fragment="")
        url = o.geturl()
    else:
        url = urljoin(host_url, url)
    return url


def checkDomain(domain, domain_urlcount):
    if domain in domain_urlcount:
        domain_urlcount[domain] += 1
    else:
        domain_urlcount[domain] = 1
    return domain_urlcount


def match_keywords(frontier):
    keyword_scores = []
    urls = [i.url for i in frontier]

    for url in urls:
        o = urlparse(url)
        path = o.path.lower()
        score, total_match = 0.0, 0
        if len(path) > 0 and path[0] == '/':
            path = path[1:]
            urlwords = set(re.split('/|_', path))
            for urlword in urlwords:
                if urlword in model.vocab:
                    for keyword in KEYWORDS:
                        similarity = model.similarity(urlword, keyword)
                        score += similarity/len(urlwords)
                        total_match += 1
        keyword_scores.append(score/max(1,total_match))
    return keyword_scores


def sortFrontier(frontier,  domainurl_count):
    inlinks_count = [len(i.inlinks) for i in frontier]
    inlinks_score = normalize(inlinks_count)

    domain_count = [domainurl_count[get_domain(i.url)] for i in frontier]
    domain_score = normalize(domain_count)

    keyword_score = match_keywords(frontier)

    for idx,obj in enumerate(frontier):
        final_score = inlinks_score[idx] + domain_score[idx] + keyword_score[idx]
        obj.score = final_score
    sortedList = sorted(frontier, key=lambda obj: obj.score, reverse=True)

    logging.info("Returning sorted list of objects (top 30){}".format([(obj.url, obj.score) for obj in sortedList[:30]]))
    return sortedList


def read_seeds(seedfile):
    logging.info("Reading the seeds")
    frontier_map = {}  # url: url object
    frontier = {}      # waveno : list of url objects
    domain_urlcount = {}
    wave_no = 1
    with open(seedfile, "r") as f:
        for line in f:
            line = line.strip()
            url = clean_url(line)
            if validators.url(url) is True:
                obj = URL(url, wave_no)
                frontier_map[url] = obj
                if wave_no not in frontier:
                    frontier[wave_no] = []
                frontier[wave_no].append(obj)

                domain_urlcount = checkDomain(get_domain(obj.url), domain_urlcount)

    return frontier_map, frontier, domain_urlcount


def main(args):
    logging.info("In main")
    # Reading seed urls
    current_wave, crawled = 1, 0
    frontier_map, frontier, domain_urlcount= read_seeds(args.seeds)
    links_crawled = set()
    id_to_url = {}
    last_visited = None

    if args.load_freq >= 0:
        frontier_map, frontier, links_crawled, id_to_url, current_wave, domain_urlcount = load_checkpoint(args.ckp, args.load_freq)
        crawled = len(links_crawled)
        logging.info("length of frontier while loading and stored current_wave: {},{}".format(len(frontier), current_wave))

    while crawled <= args.max_crawl and current_wave <= len(frontier):
        if len(frontier[current_wave]) == 0:
            # no more links to crawl from this wave
            current_wave += 1
            if current_wave in frontier:
                logging.info("Current wave no and length of frontier {},{}".format(current_wave, len(frontier[current_wave])))
                frontier[current_wave] = sortFrontier(frontier[current_wave], domain_urlcount)

            continue

        if len(links_crawled) % args.save_freq == 0:
            logging.info("length of frontier while saving: {}".format(len(frontier)))
            save_checkpoint(args.ckp, frontier, frontier_map, id_to_url, links_crawled, current_wave, domain_urlcount)

        current_obj = frontier[current_wave].pop(0)
        logging.info("Crawling at wave no {}, # of urls left {}".format(current_wave, len(frontier[current_wave])))
        logging.info("Crawling url : {}".format(current_obj.url))
        domain = get_domain(current_obj.url)

        #Checking politenesss
        isAllowed, delay = read_robotfile(current_obj.url, domain)

        if domain == last_visited and delay == 0:
            delay = 1
        last_visited = domain
        logging.info("delay {}".format(delay))

        if isAllowed:
            scrape_return = scrape_page(current_obj.url, delay)
            if scrape_return is not None:
                url, title, httpheader, content, raw_html, outlinks = scrape_return
                links_crawled.add(current_obj.url)
                id_to_url[len(id_to_url)+1] = current_obj.url
                write_to_pickle(args.cdp, len(id_to_url), current_obj.url, title, httpheader, content, raw_html)
                crawled += 1
                current_obj.update_outlinks(outlinks)

                #cleaning and storing the outlinks in next wave
                for link in outlinks:
                    link = clean_url(link, current_obj.url)
                    if validators.url(link) is True:
                        if link not in frontier_map:
                            new_obj = URL(link, current_obj.waveno + 1)
                            frontier_map[link] = new_obj
                            current_obj.update_inlink(link)

                            if current_wave+1 not in frontier:
                                frontier[current_wave+1] = []

                            frontier[current_wave + 1].append(new_obj)
                            domain_urlcount = checkDomain(get_domain(new_obj.url), domain_urlcount)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--dir", type=str, default="./test", help="")
    parser.add_argument("--seeds", type=str, default="seeds.txt", help="")
    parser.add_argument("--save_freq", type=int, default=400, help="")
    parser.add_argument("--load_freq", type=int, default=-1, help="")
    parser.add_argument("--max_crawl", type=int, default=40000, help="")
    args = parser.parse_args()

    # additional parse option
    args.ckp = os.path.join(args.dir,"checkpoint") #ckp = checkpoint
    args.cdp = os.path.join(args.dir, "crawled") #cdp = crawled data path
    args.logfile = os.path.join(args.dir, "log.txt")

    logging.basicConfig(filename=args.logfile, format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main(args)

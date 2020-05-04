from urllib.parse import urlparse, ParseResult, urljoin
import logging
import numpy as np
import os
import pickle
logger = logging.getLogger(__name__)

def get_domain(url):
    o = urlparse(url)
    domain = o.scheme + '://' + o.netloc
    return domain

def normalize(x, eps=1e-6):
    x = np.array(x)
    return (x - min(x))/(max(x) - min(x) + eps)


def save_checkpoint(ckp, frontier, frontier_map, id_to_url, links_crawled, current_wave, domain_urlcount):
    #pdb.set_trace()
    checkpoint_fname = os.path.join(ckp, "checkpoint.%d" %len(links_crawled))
    logging.info("saving checkpoint at : {}".format(str(checkpoint_fname)))

    pickle.dump(frontier, open("%s.frontier.pt"%checkpoint_fname, "wb"))
    pickle.dump(frontier_map, open("%s.frontier_map.pt"%checkpoint_fname, "wb"))
    pickle.dump(id_to_url, open("%s.id_to_url.pt"%checkpoint_fname, "wb"))
    pickle.dump(links_crawled, open("%s.links_crawled.pt"%checkpoint_fname, "wb"))
    pickle.dump(current_wave, open("%s.current_wave.pt"%checkpoint_fname, "wb"))
    pickle.dump(domain_urlcount, open("domains.nb", "wb"))


def load_checkpoint(ckp, checkpoint_iter):
    checkpoint_fname = os.path.join(ckp, "checkpoint.%d" % checkpoint_iter)
    logging.info("Loading checkpoint at : {}".format(str(checkpoint_fname)))

    frontier = pickle.load(open("%s.frontier.pt"%checkpoint_fname, "rb"))
    frontier_map = pickle.load(open("%s.frontier_map.pt"%checkpoint_fname, "rb"))
    id_to_url = pickle.load(open("%s.id_to_url.pt"%checkpoint_fname, "rb"))
    links_crawled = pickle.load(open("%s.links_crawled.pt"%checkpoint_fname, "rb"))
    current_wave = pickle.load(open("%s.current_wave.pt"%checkpoint_fname, "rb"))
    domain_urlcount = pickle.load(open("domains.nb", "rb"))

    return frontier_map, frontier, links_crawled, id_to_url, current_wave, domain_urlcount
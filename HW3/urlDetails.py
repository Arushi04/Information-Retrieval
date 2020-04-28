class URL:

    #init method or constructor
    def __init__(self, url, waveno):
        self.url = url
        self.waveno = waveno
        self.inlinks = []
        self.outlinks = []
        self.score = 0

    def update_outlinks(self,urls):
        self.outlinks = urls


    def update_inlink(self, url):
        self.inlinks.append(url)

    def update_score(self, score):
        self.score = score


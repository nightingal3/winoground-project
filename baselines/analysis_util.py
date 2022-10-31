class WinogroundResult:
    """
    Instantiates a new WinogroundResult from a string

    :param
    s: string of the format "id,tag,secondary_tag,num_main_preds,collapsed_tag,C0I0,C0I1,C1I0,C1I1"
    sep: the separator for string s, by default \xa0
    annotations: whether s comes with our annotations
    """
    def __init__(self, s, sep="\xa0", annotations=False):
        if not annotations:
            [self.id, self.tag, self.secondary_tag, self.num_main_preds, self.collapsed_tag, self.c0i0, self.c0i1, self.c1i0, self.c1i1] = s.strip().split(sep)
        else:
            [self.id, self.tag, self.secondary_tag, self.num_main_preds, self.collapsed_tag, self.c0i0, self.c0i1,
             self.c1i0, self.c1i1, self.unnatural, self.counting, self.spatial, self.commonsense, self.figurative] = s.strip().split(sep)
            self.annotations = []
            if self.unnatural == "1":
                self.annotations.append("unnatural")
            if self.counting == "1":
                self.annotations.append("counting")
            if self.unnatural == "1":
                self.annotations.append("spatial")
            if self.unnatural == "1":
                self.annotations.append("commonsense")
            if self.unnatural == "1":
                self.annotations.append("figurative")
        self.id = int(self.id)
        # since sometimes they have "Tag1, Tag2"
        self.tag = [s.strip() for s in self.tag.split(",")]
        self.secondary_tag = [s.strip() for s in self.secondary_tag.split(",")]
        self.collapsed_tag = [s.strip() for s in self.collapsed_tag.split(",")]

        self.num_main_preds = int(self.num_main_preds)
        self.c0i0 = float(self.c0i0)
        self.c0i1 = float(self.c0i1)
        self.c1i0 = float(self.c1i0)
        self.c1i1 = float(self.c1i1)

    """
    Gets the comparisons for each image and caption
    
    :return
    comparisons: a tuple (c0, c1, i0, i1) where c0 is 0 if caption 0 prefers image 0 and 1 otherwise, same for c1, 
    and i0 is 0 if image 0 prefers caption 0 and 1 otherwise, same for i1.
    Note that text score is 1 iff i0==0 and i1==1, image score is 1 iff c0==0 and c1==1, group score is 1 iff
    text score is 1 and image score is 1
    """
    def get_comparisons(self):
        c0 = 0 if self.c0i0 > self.c0i1 else 1
        c1 = 0 if self.c1i0 >= self.c1i1 else 1 # >= here to be consistent with the formal definition
        i0 = 0 if self.c0i0 > self.c1i0 else 1
        i1 = 0 if self.c0i1 >= self.c1i1 else 1 # >= here to be consistent with the formal definition
        return (c0, c1, i0, i1)

class WinogroundResultList:
    """
    Instantiates a list of WinogroundResults from a file

    :param
    filepath: path to the file to read. the file format should be a WinogroundResult string on each line, by default None
    header: whether or not the file contains a header line, by default True
    sep: the separator used in each line, by default \xa0
    annotations: whether the file comes with our annotations
    """
    def __init__(self, filepath=None, header=True, sep="\xa0", annotations=False):
        self.results = []
        if filepath is not None:
            f = open(filepath)
            lines = f.readlines()
            if header:
                lines = lines[1:]
            for line in lines:
                self.results.append(WinogroundResult(line, sep=sep, annotations=annotations))
            f.close()

    """
    Manually set the results list
    
    :param
    results: a list of WinogroundResult
    """
    def set_results(self, results):
        self.results = results

    """
    Return a new WinogroundResultList with only the desired tags. The tags can be found in the tag, secondary_tag,
    and collapsed_tag fields.
    
    :param
    tags: a list of tags to filter on
    
    :return
    new_wrl: a new WinogroundResultList with only the desired tags.
    """
    def filter_tag(self, tags):
        new_wrl = WinogroundResultList()
        new_wrl_results = set()
        for tag in tags:
            for result in self.results:
                if tag in result.tag or tag in result.secondary_tag or tag in result.collapsed_tag:
                    new_wrl_results.add(result)
        new_wrl.set_results(list(new_wrl_results))
        return new_wrl

    def filter_annotation(self, annotation):
        new_wrl = WinogroundResultList()
        new_wrl_results = set()
        for result in self.results:
            if annotation in result.annotations:
                new_wrl_results.add(result)
        new_wrl.set_results(list(new_wrl_results))
        return new_wrl

    def text_score(self):
        denom = len(self.results)
        correct = 0
        for result in self.results:
            (c0, c1, i0, i1) = result.get_comparisons()
            if i0 == 0 and i1 == 1:
                correct += 1
        return correct / denom

    def image_score(self):
        denom = len(self.results)
        correct = 0
        for result in self.results:
            (c0, c1, i0, i1) = result.get_comparisons()
            if c0 == 0 and c1 == 1:
                correct += 1
        return correct / denom

    def group_score(self):
        denom = len(self.results)
        correct = 0
        for result in self.results:
            (c0, c1, i0, i1) = result.get_comparisons()
            if i0 == 0 and i1 == 1 and c0 == 0 and c1 == 1:
                correct += 1
        return correct / denom
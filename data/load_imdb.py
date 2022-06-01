_DESCRIPTION = """\
Large Movie Review Dataset.
This is a dataset for binary sentiment classification containing substantially \
more data than previous benchmark datasets. We provide a set of 25,000 highly \
polar movie reviews for training, and 25,000 for testing. There is additional \
unlabeled data for use as well.\
"""

_CITATION = """\
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""
import scandir

class IMDBLoader():

  def _get_reviews(self, dir):
    review_files = [f.name for f in scandir.scandir(dir)]
    review_list = []
    for review_file in review_files:
        with open(dir+'/'+review_file, "r", encoding="utf8") as f:
            text = f.read()
            text = text.rstrip('\n')
        review_list.append(text)
    return review_list


  def get_data(self, base_dir):

    neg = base_dir + '/neg'
    pos = base_dir + '/pos'

    neg_review_list = self._get_reviews(neg)
    pos_review_list = self._get_reviews(pos)
    review_list = neg_review_list + pos_review_list

    # Target labels
    labels = [0]*len(neg_review_list) + [1]*len(pos_review_list)

    return review_list, labels
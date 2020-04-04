#!/usr/bin/python3
"#!/usr/bin/python3 -OO"

"""
"""

# @todo fill in the top-level doc string

###########
# Imports #
###########

from torchtext.data import TabularDataset

##############################
# Data Loading Functionality #
##############################

tv_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("comment_text", TEXT), ("toxic", LABEL),
                 ("severe_toxic", LABEL), ("threat", LABEL),
                 ("obscene", LABEL), ("insult", LABEL),
                 ("identity_hate", LABEL)]

train_data, validation_data = TabularDataset.splits(
    path="data", # the root directory where the data lies
    train='train.csv', validation="valid.csv",
    format='csv',
    skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
    fields=tv_datafields)

test_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
                   ("comment_text", TEXT)]
test = TabularDataset(
    path="data/test.csv", # the file path
    format='csv',
    skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
    fields=tst_datafields)

###############
# Main Driver #
###############

def main() -> None:
    return

if __name__ == '__main__':
    main()

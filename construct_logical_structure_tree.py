
# https://stanfordnlp.github.io/stanza/constituency.html
# https://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html


''' logical relation connectives '''

conjuction = ['and', 'as well as', 'as well', 'also', 'separately']
alternative = ['or', 'either', 'instead', 'alternatively', 'else', 'nor', 'neither']
restatement = ['specifically', 'particularly', 'in particular', 'besides', 'additionally', 'in addition', 'moreover',
               'furthermore', 'further', 'plus', 'not only', 'indeed', 'in other words', 'in fact', 'in short',
               'in the end', 'overall', 'in sum', 'in summary', 'in detail', 'in details']
instantiation = ['for example', 'for instance', 'such as', 'including', 'as an example', 'for one thing']

contrast = ['but', 'however', 'yet', 'while', 'unlike', 'rather', 'rather than', 'in comparison', 'by comparison',
            'on the other hand', 'on the contrary', 'contrary to', 'in contrast', 'by contrast', 'still', 'whereas',
            'conversely', 'not', 'no', 'none', 'nothing', 'n\'t']
concession = ['although', 'though', 'despite', 'despite of', 'in spite of', 'regardless', 'regardless of', 'whether',
              'nevertheless', 'nonetheless', 'even if', 'even though', 'even as', 'even when', 'even after',
              'even so', 'no matter']
analogy = ['likewise', 'similarly', 'as if', 'as though', 'just as', 'just like', 'namely']

temporal = ['during', 'before', 'after', 'when', 'as soon as', 'then', 'next', 'until', 'till', 'meanwhile', 'in turn',
            'meantime', 'afterwards', 'afterward', 'simultaneously', 'at the same time', 'beforehand', 'previous',
            'previously', 'earlier', 'later', 'thereafter', 'finally', 'ultimately', 'eventually', 'subsequently']

condition = ['if', 'as long as', 'unless', 'otherwise', 'except', 'whenever', 'whichever', 'provided', 'once',
             'only if', 'only when', 'depend on', 'depends on', 'depending on', 'in case']
causal = ['because', 'cause', 'as a result', 'result in', 'due to', 'therefore', 'hence', 'thus', 'thereby', 'since',
          'now that', 'consequently', 'in consequence', 'in order to', 'so as to', 'so that', 'so', 'as', 'why', 'for',
          'accordingly', 'given', 'turn out', 'turns out']

all_keywords = conjuction + alternative + restatement + instantiation + contrast + concession + analogy + temporal + condition + causal




''' construct logical structure tree '''


import stanza
from nltk.tree import Tree, ParentedTree

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')



def recover_text_from_tree(constituency_tree):
    ''' extract the text from the given constituency tree '''

    '''
    word_list = []

    def dfs(tree): # for the Tree object in nltk.tree
        if type(tree) == str:
            word_list.append(tree)
        else:
            for child_i in range(len(tree)):
                dfs(tree[child_i])

    dfs(constituency_tree)

    text = " ".join(word_list)
    '''

    text = " ".join(constituency_tree.leaves())

    return text



def match_keyword(constituency_tree, previous_matched_keyword_list):
    ''' from top-down and left-right traverse the constituency tree, find the leftmost longest matched keyword '''

    matched_flag = 0
    matched_keyword = ""
    matched_subtree = None

    for subtree in constituency_tree.subtrees():
        text_of_subtree = recover_text_from_tree(subtree)
        if text_of_subtree in all_keywords:
            presented_flag = 0
            for previous_matched_keyword in previous_matched_keyword_list:
                if text_of_subtree in previous_matched_keyword: # the matched keyword is part of or equal to the previous matched keyword
                    presented_flag = 1
                    break

            if presented_flag == 0:
                matched_flag = 1
                matched_keyword = text_of_subtree
                matched_subtree = subtree
                break

    return matched_flag, matched_keyword, matched_subtree



def extract_arguments(matched_keyword, matched_subtree):
    ''' extract the left and right argument of the matched keyword '''

    parent_text = recover_text_from_tree(matched_subtree.parent())

    right_argument = parent_text.split(matched_keyword)[-1]

    if parent_text.split(matched_keyword)[0] != "": # parent = alpha + keyword + betta
        left_argument = parent_text.split(matched_keyword)[0]
    else: # parent = keyword + betta
        if matched_subtree.parent().parent() is not None:
            grandparent_text = recover_text_from_tree(matched_subtree.parent().parent())
            if grandparent_text.split(parent_text)[0] == "":
                if matched_subtree.parent().parent().label()[:5] == "sent_":
                    if matched_subtree.parent().parent().left_sibling() is not None: # exist previous sentence
                        left_argument = recover_text_from_tree(matched_subtree.parent().parent().left_sibling())
                    else:
                        left_argument = ""
                else:
                    left_argument = ""
            else:
                left_argument = grandparent_text.split(parent_text)[0]
        else:
            left_argument = ""

    if len(right_argument) != 0: # delete starting and ending empty space
        if right_argument[0] == " ":
            right_argument = right_argument[1:]
        if right_argument[-1] == " ":
            right_argument = right_argument[:-1]

    if len(left_argument) != 0: # delete starting and ending empty space
        if left_argument[0] == " ":
            left_argument = left_argument[1:]
        if left_argument[-1] == " ":
            left_argument = left_argument[:-1]

    return left_argument, right_argument



def construct_logical_structure_tree(text):
    ''' construct the logical structure tree for the given text '''

    doc = nlp(text.lower())
    constituency_tree_string = "(Root"

    for sent_i in range(len(doc.sentences)):
        sentence_tree = doc.sentences[sent_i].constituency
        sentence_tree.label = "sent_" + str(sent_i)
        constituency_tree_string += " " + str(sentence_tree)

    constituency_tree_string += ")"
    constituency_tree = ParentedTree.fromstring(constituency_tree_string)


    logical_structure_tree = []
    # the logical relations in each sentence are saved from macro to micro perspective

    for sent_i in range(len(constituency_tree)):
        sentence_text = recover_text_from_tree(constituency_tree[sent_i])
        logical_structure_this_sentence = {"sentence_id": sent_i, "sentence_text": sentence_text, "logical_relation": []}

        # initialize
        previous_matched_keyword_list = []
        matched_flag = 1

        # recursively match keyword from the sub constituency_tree that corresponds to the betta argument
        while matched_flag != 0:
            matched_flag, matched_keyword, matched_subtree = match_keyword(constituency_tree[sent_i], previous_matched_keyword_list)
            if matched_flag == 1:
                previous_matched_keyword_list.append(matched_keyword)
                left_argument, right_argument = extract_arguments(matched_keyword, matched_subtree)
                logical_structure_this_sentence["logical_relation"].append({"logical_keyword": matched_keyword, "left_argument": left_argument, "right_argument": right_argument})

        logical_structure_tree.append(logical_structure_this_sentence)

    return logical_structure_tree





''' try some examples '''

text = "example sentence"
logical_structure_tree = construct_logical_structure_tree(text)
print(logical_structure_tree)















# stop here

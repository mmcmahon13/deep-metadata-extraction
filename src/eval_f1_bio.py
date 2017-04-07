from __future__ import division
from __future__ import print_function
import time
import numpy as np
import sys


# compute various F1 statistics given a numpy array confusion matrix, and the index of the "outside" class
# if outside_idx=-1, there is no outside class
def compute_f1(confusion, label_map, gold_ax, outside_idx=-1):
    pred_ax = 1 if gold_ax == 0 else 0
    o_mask = np.zeros(len(label_map))
    if outside_idx > -1:
        o_mask[outside_idx] = 1

    tps = np.diag(confusion)
    tpfps = np.sum(confusion, axis=int(not pred_ax))
    tpfns = np.sum(confusion, axis=int(not gold_ax))

    masked_tps = np.ma.masked_array(tps, o_mask)
    masked_tpfps = np.ma.masked_array(tpfps, o_mask)
    masked_tpfns = np.ma.masked_array(tpfns, o_mask)

    # is this correct? shouldn't all_correct be tps + tns?
    all_correct = np.sum(tps)
    total = np.sum(confusion)
    accuracy = all_correct/total

    precisions = tps/tpfps
    recalls = tps/tpfns
    f1s = 2*precisions*recalls/(precisions+recalls)

    masked_precisions = np.ma.masked_array(precisions, o_mask)
    masked_recalls = np.ma.masked_array(recalls, o_mask)

    precision_macro = np.ma.mean(masked_precisions)
    recall_macro = np.ma.mean(masked_recalls)
    f1_macro = 2*precision_macro*recall_macro/(precision_macro+recall_macro)

    masked_tps_total = np.ma.sum(masked_tps)
    precision_micro = masked_tps_total/np.ma.sum(masked_tpfps)
    recall_micro = masked_tps_total/np.ma.sum(masked_tpfns)
    f1_micro = 2*precision_micro*recall_micro/(precision_micro+recall_micro)

    print("\t%10s\tPrec\tRecall\tAccuracy" % ("F1"))
    print("%10s\t%2.2f\t%2.2f\t%2.2f\t%2.2f" % ("Micro (Tok)", f1_micro*100, precision_micro*100, recall_micro*100, accuracy*100))
    print("%10s\t%2.2f\t%2.2f\t%2.2f" % ("Macro (Tok)", f1_macro*100, precision_macro*100, recall_macro*100))
    print("----------")
    for label in label_map:
        idx = label_map[label]
        if not idx == outside_idx:
            print("%10s\t%2.2f\t%2.2f\t%2.2f" % (label, f1s[idx]*100, precisions[idx]*100, recalls[idx]*100))
    sys.stdout.flush()


def token_eval(batches, predictions, label_map, type_int_int_map, outside_idx, pad_width, extra_text=""):
    num_types = len(label_map)
    confusion = np.zeros((num_types, num_types))
    if extra_text != "":
        print(extra_text)
    for predictions, (label_batch, token_batch, shape_batch, char_batch, seq_len_batch, mask_batch) in zip(predictions, batches):
        for preds, labels, seq_lens in zip(predictions, label_batch, seq_len_batch):
            start = pad_width
            for seq_len in seq_lens:
                for i in range(seq_len):
                # for pred, label in zip(preds[pad_width:seq_len+pad_width], labels[pad_width:seq_len+pad_width]):
                    # this will give you token-level F1
                    confusion[type_int_int_map[preds[i+start]], type_int_int_map[labels[i+start]]] += 1
                start += 2*pad_width + seq_len
    compute_f1(confusion, label_map, gold_ax=0, outside_idx=outside_idx)


def is_start(curr):
    return curr[0] == "B" or curr[0] == "U"


def is_continue(curr):
    return curr[0] == "I" or curr[0] == "L"


def is_background(curr):
    return not is_start(curr) and not is_continue(curr)


def is_seg_start(curr, prev):
    return (is_start(curr) and not is_continue(curr)) or (is_continue(curr) and (prev is None or is_background(prev) or prev[1:] != curr[1:]))

def segment_eval(batches, predictions, label_map, type_int_int_map, labels_id_str_map, vocab_id_str_map, outside_idx, pad_width, start_end, extra_text="", verbose=False):
    # label_map will contain the broken up types
    # the rest of the maps will not

    if extra_text != "":
        print(extra_text)

    def print_context(width, start, tok_list, pred_list, gold_list):
        for offset in range(-width, width+1):
            idx = offset + start
            if 0 <= idx < len(tok_list):
                print("%s\t%s\t%s" % (vocab_id_str_map[tok_list[idx]], labels_id_str_map[pred_list[idx]], labels_id_str_map[gold_list[idx]]))
        print()

    # TODO: there needs to be a key for each segment (not just the heirarchical mashed-together versions)
    # index the count maps by type string, rather than by some int id (because type_int_int_map will be messed up)
    pred_counts = {s: 0 for s, t in label_map.items()}
    gold_counts = {s: 0 for s, t in label_map.items()}
    correct_counts = {s: 0 for s, t in label_map.items()}
    # pred_counts = {}
    # gold_counts = {}
    # correct_counts = {}
    # # split up labels and add them to the count dicts
    # print("creating label counts maps")
    # for label,t in label_map.items():
    #     label_split = label.split('/')
    #     for s in label_split:
    #         print(s)
    #         pred_counts[s] = 0
    #         gold_counts[s] = 0
    #         correct_counts[s] = 0

    token_count = 0
    boundary_viols = 0
    type_viols = 0

    stop = 0
    num_citations = 0

    # iterate over batches
    for predictions, (dev_label_batch, dev_token_batch, dev_shape_batch, dev_char_batch, dev_seq_len_batch, dev_tok_len_batch, mask_batch) in zip(predictions, batches):
        # # TODO: for debug: stop after first batch
        # if stop > 0:
        #     break
        # stop += 1
        # print(mask_batch[0])
        # iterate over examples in batch
        for preds, labels, tokens, seq_lens in zip(predictions, dev_label_batch, dev_token_batch, dev_seq_len_batch):
            # if num_citations < 3:
            num_citations += 1
            start = pad_width
            # print(seq_lens)
            # print(start)
            # print(len(seq_lens))
            # print(len(preds))
            # print("sum seq lens", sum(seq_lens))
            # print("sum seq lens + pad", sum(seq_lens) + 2*(seq_lens != 0).sum()*pad_width)
            # print(preds)
            # print(labels)

            # lengths of sentences (citation in this case)
            for seq_len in seq_lens:
                # skip over the <PAD> tags?
                # these are only integers
                predicted = preds[start:seq_len+start]
                golds = labels[start:seq_len+start]
                toks = tokens[start:seq_len+start]
                # print("start: ", start)
                # print("len predicted: ", len(predicted))
                # print("predicted:")
                # print(predicted)
                # print(map(lambda p: labels_id_str_map[p], predicted))
                # print("gold:")
                # print(golds)
                # print(map(lambda p: labels_id_str_map[p], golds))

                # TODO: chop up predictions into non-heirarchical segment labels, iterate over indices of those
                # holds the split up type strings
                predicted_split = [None for p in predicted]
                golds_split = [None for g in golds]
                max_num_types = 0
                # print("pred")
                for p, pred in enumerate(predicted):
                    pred_str = labels_id_str_map[pred]
                    pred_split = pred_str.split("/")
                    # print(pred_split)
                    if len(pred_split) > max_num_types:
                        max_num_types = len(pred_split)
                    predicted_split[p] = pred_split

                # print("gold")
                for g, gold in enumerate(golds):
                    gold_str = labels_id_str_map[gold]
                    gold_split = gold_str.split("/")
                    # print(gold_split)
                    if len(gold_split) > max_num_types:
                        max_num_types = len(gold_split)
                    golds_split[g] = gold_split

                # pad the label lists so that they all have the same length
                c = 0
                # put in a bunch of dummy values such that gold and predicted will never match, even if they both have dummy values?
                for p in predicted_split:
                    while len(p) < max_num_types:
                        p.append("O-<DUMMY>"+str(c))
                        c += 1
                for g in golds_split:
                    while len(g) < max_num_types:
                        g.append("O-<DUMMY>"+str(c))
                        c += 1

                for p in predicted_split:
                    assert len(p) == max_num_types

                for g in golds_split:
                    assert len(g) == max_num_types

                # go through the tokens in a sequence
                # print("citation length: ", seq_len)
                # check segments at each level of the heirarchy
                for e in range(max_num_types):
                    for i in range(seq_len):
                        token_count += 1
                        pred = predicted[i] # whole label int id
                        gold = golds[i] # whole label int id
                        # get the integer values for the predicted label and the gold label, then get the corresponding string labels
                        pred_str = predicted_split[i][e]
                        gold_str = golds_split[i][e]
                        # print("predicted: ", pred_str, "gold: ", gold_str)
                        # get gold and predicted string labels for the previous token
                        gold_prev = None if i == 0 else golds_split[i-1][e]
                        pred_prev = None if i == 0 else predicted_split[i-1][e]
                        # get gold and predicted types without BIO labels
                        pred_type = pred_str[2:]
                        gold_type = gold_str[2:]
                        # check whether the predicted or gold label begins a segment, increment the number of predictions for that type if so
                        pred_start = False
                        gold_start = False
                        # this will never be true for two dummy values
                        if is_seg_start(pred_str, pred_prev):
                            # print("incrementing ", pred_type, " for predicted ")
                            pred_counts[pred_type] += 1
                            pred_start = True
                        if is_seg_start(gold_str, gold_prev):
                            # print("incrementing ", pred_type, " for gold ")
                            gold_counts[gold_type] += 1
                            gold_start = True

                        # if both predicted and gold labels start a new segment: TODO what is this actually doing?
                        # will never fire for two dummy values, since they don't start segments
                        if pred_start and gold_start:
                            # check for type violation (ignore this, we don't care)
                            if pred_type != gold_type:
                                # i = sentence length
                                j = i + 1
                                stop_search = False
                                # look through the rest of the sentence for other violations at that type index
                                while j < seq_len and not stop_search:
                                    pred2 = predicted_split[j][e]
                                    gold2 = predicted_split[j][e]
                                    # pred_type2 = type_int_int_map[predicted[j]]
                                    pred_type2 = pred2[2:]
                                    pred_continue = is_continue(pred2)
                                    gold_continue = is_continue(gold2)

                                    # if the remaining tokens' prediction or gold labels aren't continuations
                                    # or if the current token's predicted type doesn't match the original token's gold type
                                    # or if we're at the end of the sentence
                                    if not pred_continue or not gold_continue or pred_type2 != gold_type or j == seq_len - 1:
                                        # check for type violation
                                        if pred_continue and gold_continue and pred_type2 != pred_type:
                                            type_viols += 1
                                            if verbose:
                                                print_context(2, j, toks, predicted, golds)
                                            stop_search = True
                                    j += 1
                                    # type_viols += 1
                                    # if verbose:
                                    #     print_context(2, i, toks, predicted, golds)
                                    # if there's no type violation:
                            else: # if pred_type and gold_type are the same...
                                # if we're at the end of the sentence we got the segment right?
                                if i == seq_len - 1:
                                    # print("correct prediction: pred: ", pred_type, "gold", gold_type)
                                    correct_counts[gold_type] += 1
                                else:
                                # if not, check the rest of the segment for violations:
                                    j = i + 1
                                    stop_search = False
                                    while j < seq_len and not stop_search:
                                        pred2 = predicted_split[j][e]
                                        gold2 = predicted_split[j][e]
                                        # pred_type2 = type_int_int_map[predicted[j]]
                                        pred_type2 = pred2[2:]
                                        pred_continue = is_continue(pred2)
                                        gold_continue = is_continue(gold2)

                                        if not pred_continue or not gold_continue or pred_type2 != gold_type or j == seq_len - 1:
                                            # check for type violation
                                            if pred_continue and gold_continue and pred_type2 != pred_type:
                                                type_viols += 1
                                                if verbose:
                                                    print_context(2, j, toks, predicted, golds)
                                            # check for boundary violation
                                            if pred_continue != gold_continue:
                                                # I or L must come after B or I
                                                if pred_continue:
                                                    last_bilou = labels_id_str_map[predicted[j-1]][0]
                                                    if last_bilou != "B" and last_bilou != "I":
                                                        boundary_viols += 1
                                                        if verbose:
                                                            print_context(2, j, toks, predicted, golds)
                                                if pred2[0] == "B":
                                                    next_bilou = labels_id_str_map[predicted[j+1]][0] if j+1 < seq_len else "I" # wrong
                                                    if next_bilou != "I" and next_bilou != "L":
                                                        boundary_viols += 1
                                                        if verbose:
                                                            print_context(2, j, toks, predicted, golds)

                                            # if pred_continue == gold_continue:
                                            if (not pred_continue and not gold_continue) or (pred_continue and gold_continue and pred_type2 == gold_type):
                                                # print("correct prediction: pred: ", pred_type, "gold", gold_type)
                                                correct_counts[gold_type] += 1
                                            stop_search = True
                                        j += 1
                        # check for boundary violation
                        # if one of the things doesn't start a new segment
                        elif pred_start != gold_start:
                                # pred_str will never be a continue if it's a dummy
                                # I or L must come after B or I
                                if is_continue(pred_str):
                                    if i > 0:
                                        last_bilou = pred_prev[0]
                                        if last_bilou != "B" and last_bilou != "I":
                                            boundary_viols += 1
                                            if verbose:
                                                print_context(2, i, toks, predicted, golds)
                                    else:
                                        boundary_viols += 1
                                # pred_str will never be B if it is a dummy
                                if pred_str[0] == "B":
                                    if i < seq_len-1:
                                        next_bilou = labels_id_str_map[predicted[i+1]][0]
                                        if next_bilou != "I" and next_bilou != "L":
                                            boundary_viols += 1
                                            if verbose:
                                                print_context(2, i, toks, predicted, golds)
                                    else:
                                        boundary_viols += 1
                                        if verbose:
                                            print_context(2, i, toks, predicted, golds)
            start += seq_len + (2 if start_end else 1)*pad_width

    for key, value in pred_counts.items():
        print(key, ": predicted counts: ", value, " correct counts: ", correct_counts[key], " gold counts: ", gold_counts[key])
    print("\n")


    all_correct = np.sum([p if i not in outside_idx else 0 for i, p in enumerate(correct_counts.values())])
    all_pred = np.sum([p if i not in outside_idx else 0 for i, p in enumerate(pred_counts.values())])
    all_gold = np.sum([p if i not in outside_idx else 0 for i, p in enumerate(gold_counts.values())])

    # precisions = [correct_counts[i] / pred_counts[i] if pred_counts[i] != 0 else 0.0 for i in pred_counts.keys()]
    # recalls = [correct_counts[i] / gold_counts[i] if gold_counts[i] != 0 else 1.0 for i in gold_counts.keys()]
    # f1s = [2 * precision * recall / (recall + precision) if recall + precision != 0 else 0.0 for precision, recall in
    #        zip(precisions, recalls)]

    precisions = {i: correct_counts[i] / pred_counts[i] if pred_counts[i] != 0 else 0.0 for i in pred_counts.keys()}
    recalls = {i: correct_counts[i] / gold_counts[i] if gold_counts[i] != 0 else 1.0 for i in gold_counts.keys()}
    f1s = {i: 2 * precisions[i] * recalls[i] / (recalls[i] + precisions[i]) if recalls[i] + precisions[i] != 0 else 0.0 for i in precisions.keys()}

    precision_macro = np.mean([v for v in precisions.values()])
    recall_macro = np.mean([r for r in recalls.values()])
    f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)

    precision_micro = all_correct / all_pred
    recall_micro = all_correct / all_gold
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro)

    accuracy = all_correct / all_gold

    print("\t%10s\tPrec\tRecall" % ("F1"))
    print("%10s\t%2.2f\t%2.2f\t%2.2f" % ("Micro (Seg)", f1_micro * 100, precision_micro * 100, recall_micro * 100))
    print("%10s\t%2.2f\t%2.2f\t%2.2f" % ("Macro (Seg)", f1_macro * 100, precision_macro * 100, recall_macro * 100))
    print("-------")
    for t in label_map:
        # idx = label_map[t]
        # TODO: changed this so that the types that don't occur in the dev file don't get printed out
        if gold_counts[t] != 0:
            print("%10s\t%2.2f\t%2.2f\t%2.2f" % (t, f1s[t] * 100, precisions[t] * 100, recalls[t] * 100))
    print("Processed %d tokens with %d phrases; found: %d phrases; correct: %d." % (token_count, all_gold, all_pred, all_correct))
    print("Found %d type violations, %d boundary violations." % (type_viols, boundary_viols))
    sys.stdout.flush()
    return f1_micro, precision_micro


def print_training_error(num_examples, start_time, epoch_loss, step):
    print("%20d examples at %5.2f examples/sec. Error: %5.5f" %
          (num_examples, num_examples / (time.time() - start_time), (epoch_loss / step)))
    sys.stdout.flush()

# label_map = {'LOC': 1, 'MISC': 4, 'O': 3, 'PER': 0, 'ORG': 2}
# confusion = np.array(
#     [[508, 29, 66, 53, 75],
#      [227, 1308, 227, 149, 104],
#      [143, 125, 508, 58, 59],
#      [2228, 592, 1225, 41765, 671],
#     [8, 14, 35, 12,  334]]
# )
# compute_f1(confusion, label_map, gold_ax=0, outside_idx=label_map["O"])
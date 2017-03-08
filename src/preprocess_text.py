# Text preprocessing utilities based on the pipeline provided for the BioNLP 2011 shared task
# https://github.com/ninjin/bionlp_st_2011_supporting

# based on https://github.com/saffsd/geniatagger/blob/master/tokenize.cpp
# this might not be accurate, so maybe skip it?
def tokenize(text):
    if text[0] == '"':
        text.replace(0, 1, "`` ")
    text.replace(" \"", "  `` ")
    text.replace("(\"", "( `` ")
    text.replace("[\"", "[ `` ")
    text.replace("{\"", "{ `` ")
    text.replace("<\"", "< `` ")

    text.replace("...", " ... ")

    text.replace(",", " , ")
    text.replace("", "  ")
    text.replace(":", " : ")
    text.replace("@", " @ ")
    text.replace("#", " # ")
    text.replace("$", " $ ")
    text.replace("%", " % ")
    text.replace("&", " & ")

    pos = len(text)
    while pos > 0 and text[pos] == ' ':
        pos -= 1
    while pos > 0:
        c = text[pos]
        if (c == '[' or c == ']' or c == ')' or c == '}' or c == '>' or c == '"' or c == '\''):
            pos -= 1
            continue
        break

    if text[pos] == '.' and not (pos > 0 and text[pos-1] == '.'):
        text = text[0:pos-1] + ' .' + text[pos:]

    text.replace("?", " ? ")
    text.replace("!", " ! ")

    text.replace("[", " [ ")
    text.replace("]", " ] ")
    text.replace("(", " ( ")
    text.replace(")", " ) ")
    text.replace("{", " { ")
    text.replace("}", " } ")
    text.replace("<", " < ")
    text.replace(">", " > ")

    text.replace("--", " -- ")

    # s.text.replace(string::
    #     size_type(0), 0, " ")
    # s.text.replace(s.size(), 0, " ")

    text.replace("\"", " '' ")

    text.replace("' ", " ' ", '\'')
    text.replace("'s ", " 's ")
    text.replace("'S ", " 'S ")
    text.replace("'m ", " 'm ")
    text.replace("'M ", " 'M ")
    text.replace("'d ", " 'd ")
    text.replace("'D ", " 'D ")
    text.replace("'ll ", " 'll ")
    text.replace("'re ", " 're ")
    text.replace("'ve ", " 've ")
    text.replace("n't ", " n't ")
    text.replace("'LL ", " 'LL ")
    text.replace("'RE ", " 'RE ")
    text.replace("'VE ", " 'VE ")
    text.replace("N'T ", " N'T ")

    text.replace(" Cannot ", " Can not ")
    text.replace(" cannot ", " can not ")
    text.replace(" D'ye ", " D' ye ")
    text.replace(" d'ye ", " d' ye ")
    text.replace(" Gimme ", " Gim me ")
    text.replace(" gimme ", " gim me ")
    text.replace(" Gonna ", " Gon na ")
    text.replace(" gonna ", " gon na ")
    text.replace(" Gotta ", " Got ta ")
    text.replace(" gotta ", " got ta ")
    text.replace(" Lemme ", " Lem me ")
    text.replace(" lemme ", " lem me ")
    text.replace(" More'n ", " More 'n ")
    text.replace(" more'n ", " more 'n ")
    text.replace("'Tis ", " 'T is ")
    text.replace("'tis ", " 't is ")
    text.replace("'Twas ", " 'T was ")
    text.replace("'twas ", " 't was ")
    text.replace(" Wanna ", " Wan na ")
    text.replace(" wanna ", " wanna ")
    return text

def unicode_to_ascii(text):
    pass
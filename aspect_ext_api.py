import pandas as pd
def aspects_extraction(overall_prediction, stop_words, lemmatizer, nlp):
    df = overall_prediction
    df = df.reset_index()
    df['review'] = df['review'].str.lower()

    noun_adj_pairs = {}
    all_tuples_list = []
    predictions_list = []
    
    for idx, text in enumerate(df["review"]):
        doc = nlp(text)
        for i in range(len(doc)):
            
            # each element in a sentence is a token
            token = doc[i]
            
            # extraction form: (compound) + (not) + adj + noun
            # example input: This is (not) a fantastic (barbecue) restaurant.
            # extraction result: {noun: "(barbecue) restaurant", adj: ["(not) fantastic"]}
            if (token.pos_ == "NOUN") and (token.dep_ != "compound"):
                childrenDep = [c.dep_ for c in token.children]
                compound = None
                adj_list = []
                # if compound + noun
                if "compound" in childrenDep:
                    # loop over noun's children to get compound text
                    for c in token.children:
                        if c.dep_ == 'compound':
                            compound = c.text
                            # loop over compound's children to get adj if any
                            for cc in c.children:
                                if cc.dep_ in ['amod', 'ccomp']:
                                    # tackle negation that affect verb connected to the noun, such as: is not a good meal
                                    # tackle "other" and "another"
                                    aux_ancestor = next((a for a in token.ancestors if a.pos_ == "AUX"), None)
                                    children_deps = [sb.dep_ for sb in aux_ancestor.children] if aux_ancestor is not None else []
                                    
                                    adj_list.append(cc.text if (cc.text not in ["another", "other"]) and ("neg" not in childrenDep) and ("neg" not in [gc.dep_ for gc in cc.children]) and ("neg" not in children_deps if token.dep_ == "attr" else True) else f"not {cc.text}")
                        
                        elif (c.dep_ in ['amod', 'ccomp']) and (c.text not in ["another", "other"]):
                            adj_list.append(c.text if ("neg" not in childrenDep) and ("neg" not in [gc.dep_ for gc in c.children]) and ("neg" not in [sb.dep_ for sb in next((a for a in token.ancestors if a.pos_ == "AUX"), c).children] if token.dep_ == "attr" else True) else f"not {c.text}")     
                # if regular noun
                else:            
                    for child in token.children:
                        if child.dep_ in ['amod', 'ccomp'] and child.text not in ["other", "another"]:  # and ("neg" not in [ua.dep_ for ua in a.children for a in doc[3].ancestors]
                            adj_list.append(child.text if ("neg" not in childrenDep) and ("neg" not in [gc.dep_ for gc in child.children]) and ("neg" not in [sb.dep_ for sb in next((a for a in token.ancestors if a.pos_ == "AUX"), child).children] if token.dep_ == "attr" else True) else f"not {child.text}")
                # noun and adj to pairs
                if compound is not None:
                    noun = lemmatizer.lemmatize(token.text, pos='n')
                    noun = f"{compound} {noun}"
                else:
                    noun = token.text
                    noun = lemmatizer.lemmatize(noun, pos='n')
                if adj_list:
                    #noun = wordnet_lemmatizer.lemmatize(noun, pos='n')
                    noun_adj_pairs.setdefault(noun, []).extend(adj_list)
                else:
                    pass
            
            # extraction form: (compound) + noun + be/verb + (not) + adj; (compound) + noun ... pronoun + be/verb + (not) + adj
            # example input: The (strimp) burger was (not) good. It was salty.
            # extraction result: {noun: "(strimp) burger", adj: ["(not) good", "salty"]}
            elif (token.lemma_ == "be") or (token.pos_ == "VERB"):
                for child in token.children:
                    # child = j
                    if child.dep_ in ["nsubj", "attr"] and (child.pos_ in ["NOUN", "PROPN", "PRON"]):
                        if any(p.pos_ == 'NOUN' for p in token.ancestors) and token.dep_ == 'relcl':
                            noun = list(token.ancestors)[0]
                            compound = None
                            if "compound" in [gc.dep_ for gc in noun.children if gc.dep_ == "compound"]:
                                compound = " ".join([gc.text for gc in noun.children if gc.dep_ == "compound"])
                            det = None
                            others = [gc.text for gc in noun.children if (gc.text == "another" or gc.text=='other')]
                            if "another" in others or "other" in others:
                                det = " ".join([gc.text for gc in noun.children if gc.text == "other" or gc.text == "another"])    
                            if compound is None and det is not None:
                                noun = f"{det} {noun}"
                            elif compound is not None and det is None:
                                noun = f"{compound} {noun}"
                            elif compound is None and det is None:
                                noun = f"{noun}"
                            else:
                                noun = f"{det} {compound} {noun}"
                            noun = lemmatizer.lemmatize(noun, pos='n')
                        # tackle pronoun and excluding pronoun which is "we" or "everyone"
                        elif child.pos_ == "PRON" and child.text not in ["we", "everyone"]:
                            # loop back to get noun and compound if any
                            for k in range(i, -1, -1):
                                if doc[k].pos_ == "NOUN":
                                    compound = None
                                    if "compound" in [c.dep_ for c in doc[k].children if c.dep_ == "compound"]:
                                        compound = " ".join([c.text for c in doc[k].children if c.dep_ == "compound"])
                                    det = None
                                    if "det" in [c.dep_ for c in doc[k].children if (c.text == "another" or c.text=='other')]:
                                        det = " ".join([c.text for c in doc[k].children if c.dep_ == "det"])    
                                    
                                    if compound is None and det is not None:
                                        noun = f"{det} {doc[k]}"
                                    elif compound is not None and det is None:
                                        noun = f"{compound} {doc[k]}"
                                    elif compound is None and det is None:
                                        noun = doc[k].text
                                    else:
                                        noun = f"{det} {compound} {doc[k]}"
                                    noun = lemmatizer.lemmatize(noun, pos='n')     
                                    break
                                else:                                  
                                    noun = child.text
                                    noun = lemmatizer.lemmatize(noun, pos='n')
                                    pass
                        else:
                            compound = None
                            if "compound" in [c.dep_ for c in child.children if c.dep_ == "compound"]:
                                compound = " ".join([c.text for c in child.children if c.dep_ == "compound"])
                            det = None
                            if "det" in [c.dep_ for c in child.children if (c.text == "another" or c.text=='other')]:
                                det = " ".join([c.text for c in child.children if c.dep_ == "det"])    
                            
                            if compound is None and det is not None:
                                noun = f"{det} {child}"
                            elif compound is not None and det is None:
                                noun = f"{compound} {child}"
                            elif compound is None and det is None:
                                noun = child.text
                            else:
                                noun = f"{det} {compound} {child}"
                            noun = lemmatizer.lemmatize(noun, pos='n')    

                        adj = []
                        negated = False
                        for c in token.children:
                            
                            if c.dep_ == "neg":
                                negated = True
                            if c.pos_ == "ADJ" and c.text not in ["other", "another"]:
                                if negated:
                                    adj.append("not " + c.text)
                                else:
                                    adj.append(c.text)
                                for gc in c.children:
                                    if gc.pos_ == "ADJ" and gc.text not in ["other", "another"]:
                                        if "neg" in [ggc.dep_ for ggc in gc.children]:
                                            adj.append("not " + gc.text)
                                        else:
                                            adj.append(gc.text)
                                    for ggc in gc.children:
                                        if ggc.pos_ == "ADJ" and ggc.text not in ["other", "another"]:
                                            if "neg" in [gggc.dep_ for gggc in ggc.children]:
                                                adj.append("not " + ggc.text)
                                            else:
                                                adj.append(ggc.text)
                            
                        if noun and adj:
                            if noun not in noun_adj_pairs:
                                noun_adj_pairs[noun] = []
                            for i in adj:
                                if i not in noun_adj_pairs[noun]:
                                    noun_adj_pairs[noun].append(i)
            
            # extraction form: (compound) + noun + be + (not) + meta modifier
            # example input: The (order) machine is not user-friendly.
            # extraction result: {noun: "(order) machine", adj: ["(not) user-friendly"]}                                      
            elif token.dep_ == "meta" and token.pos_ == "ADJ":
                for a in token.ancestors:
                    if a.lemma_ == "be":
                        for child in a.children:
                            if child.dep_ in ["nsubj", "attr"] and child.pos_ == "NOUN":
                                
                                noun = child.text
                                noun = lemmatizer.lemmatize(noun, pos='n')
                                compound = None
                                for gc in child.children:
                                    if gc.dep_ == "compound":
                                        compound = gc.text
                                        
                                adj = []
                                adj.append(token)
                                negated = False
                                for c in token.children:
                                    if c.dep_ == "neg":
                                        negated = True
                                    if c.pos_ == "ADJ":
                                        if negated:
                                            adj.append("not " + c.text)
                                        else:
                                            adj.append(c.text)
                                            
                                if compound is not None:
                                    noun = lemmatizer.lemmatize(noun, pos='n')
                                    noun = f"{compound} {noun}"
                                else:
                                    noun = child.text
                                    noun = lemmatizer.lemmatize(noun, pos='n')
                                if noun and adj:
                                    if noun not in noun_adj_pairs:
                                        noun_adj_pairs[noun] = []
                                    for i in adj:
                                        if i not in noun_adj_pairs[noun]:
                                            noun_adj_pairs[noun].append(i)  
        
        # wrap aspects, adjectives, and previous overall_prediction together
        tuples_list = [(k, v) for k, values in noun_adj_pairs.items() for v in values]
        all_tuples_list.extend(tuples_list)
        for _ in range(len(tuples_list)):
            predictions_list.append(df.loc[idx, "prediction"])
        noun_adj_pairs = {}
    aspects_adj = pd.DataFrame(all_tuples_list, columns=['aspect', 'adjective'])
    aspects_adj['aspect_adj'] = aspects_adj[['adjective', 'aspect']].agg(' '.join, axis=1)
    aspects_adj['predictions_overall'] = predictions_list  
    return aspects_adj

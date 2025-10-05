import heapq
import re

class PlagiarismChecker:
    
    def __init__(self, thresh=0.7):
        self.thresh = thresh
        self.nodes_checked = 0
    
    # Preprocessing text
    def Preprocessing_text(self, txt):
        # breaking into sentences
        sentences = re.split(r'[.!?]+', txt)
        processed = []
        for sent in sentences:
            sent = sent.lower() # converted lowercase
            sent = sent.strip()
            # removed punctuation marks
            sent = re.sub(r'[^\w\s]', '', sent)
            sent = re.sub(r'\s+', ' ', sent)  # extra spaces
            if sent != '':
                processed.append(sent)
        return processed
    
    # levenshtein distance calculation
    def calc_edit_dist(self, s1, s2):
        len1 = len(s1)
        len2 = len(s2)
        
        table = []
        for i in range(len1+1):
            row = []
            for j in range(len2+1):
                row.append(0)
            table.append(row)
        
        for i in range(len1+1):
            table[i][0] = i
        for j in range(len2+1):
            table[0][j] = j
        
        # calculate distances
        for i in range(1, len1+1):
            for j in range(1, len2+1):
                if s1[i-1] == s2[j-1]:
                    table[i][j] = table[i-1][j-1]  # no change needed
                else:
                    # take min of insert, delete, replace
                    table[i][j] = 1 + min(table[i-1][j], table[i][j-1], table[i-1][j-1])
        
        return table[len1][len2]
    
    # calculate similarity of two sentences 
    def similarity(self, sent1, sent2):
        if len(sent1) == 0 or len(sent2) == 0:
            return 0.0
        
        longer = len(sent1)
        if len(sent2) > longer:
            longer = len(sent2)
        
        distance = self.calc_edit_dist(sent1, sent2)
        sim_score = 1.0 - (distance / longer)
        return sim_score
    
    # heuristic for remaining cost
    def calc_heuristic(self, pos1, pos2, total1, total2):
        left1 = total1 - pos1
        left2 = total2 - pos2
        h_val = abs(left1 - left2) * 0.5
        return h_val
    
    # get cost for alignment
    def get_cost(self, idx1, idx2, sentences1, sentences2):
        if idx1 is None and idx2 is None:
            return 999999
        
        # skipping one sentence
        if idx1 is None or idx2 is None:
            return 0.5
        
        # aligning two sentences
        sim_val = self.similarity(sentences1[idx1], sentences2[idx2])
        cost = 1.0 - sim_val  # less similar = more cost
        return cost
    
    # main A* search algorithm
    def search_alignment(self, doc1, doc2):
        size1 = len(doc1)
        size2 = len(doc2)
        
        # open list with priority queue
        # storing: (f_cost, g_cost, position1, position2, alignment_path)
        open_queue = []
        start_node = (0, 0, 0, 0, [])
        heapq.heappush(open_queue, start_node)
        
        # keep track of visited states
        visited = set()
        
        # store g costs
        g_cost_map = {}
        g_cost_map[(0, 0)] = 0
        
        self.nodes_checked = 0
        
        while len(open_queue) > 0:
            current = heapq.heappop(open_queue)
            f_val, g_val, pos1, pos2, path_so_far = current
            
            if pos1 == size1 and pos2 == size2:  # reached end
                return path_so_far
            
            current_state = (pos1, pos2)
            
            if current_state in visited:
                continue
            
            visited.add(current_state)
            self.nodes_checked = self.nodes_checked + 1
            
            # generate possible next states
            next_states = []
            
            # case 1: match current sentences
            if pos1 < size1 and pos2 < size2:
                c = self.get_cost(pos1, pos2, doc1, doc2)
                next_states.append((pos1+1, pos2+1, c, (pos1, pos2)))
            
            # case 2: skip sentence from first doc
            if pos1 < size1:
                c = self.get_cost(pos1, None, doc1, doc2)
                next_states.append((pos1+1, pos2, c, (pos1, None)))
            
            # caes 3: skip sentence from second doc
            if pos2 < size2:
                c = self.get_cost(None, pos2, doc1, doc2)
                next_states.append((pos1, pos2+1, c, (None, pos2)))
            
            # check all neighbors
            for next_pos1, next_pos2, edge_cost, action in next_states:
                next_state = (next_pos1, next_pos2)
                
                if next_state in visited:
                    continue
                
                tentative_g = g_val + edge_cost
                
                # check if this is better path
                if next_state not in g_cost_map or tentative_g < g_cost_map[next_state]:
                    g_cost_map[next_state] = tentative_g
                    h_val = self.calc_heuristic(next_pos1, next_pos2, size1, size2)
                    f_val = tentative_g + h_val
                    updated_path = path_so_far + [action]
                    heapq.heappush(open_queue, (f_val, tentative_g, next_pos1, next_pos2, updated_path))
        
        return []  # no path found
    
    # detect plagiarism between two texts
    def check_plagiarism(self, text1, text2):
        sentences1 = self.Preprocessing_text(text1)
        sentences2 = self.Preprocessing_text(text2)
        
        print(f"Document 1: {len(sentences1)} sentences")
        print(f"Document 2: {len(sentences2)} sentences")
        
        # run A* to get alignment
        alignment_result = self.search_alignment(sentences1, sentences2)
        
        # analyze the alignment
        plagiarism_matches = []
        total_similarity = 0
        count_aligned = 0
        
        for alignment_pair in alignment_result:
            i, j = alignment_pair
            if i is not None and j is not None:
                sim = self.similarity(sentences1[i], sentences2[j])
                count_aligned = count_aligned + 1
                total_similarity = total_similarity + sim
                
                # check if similarity above threshold
                if sim >= self.thresh:
                    match_info = {
                        'index1': i,
                        'index2': j,
                        'sentence1': sentences1[i],
                        'sentence2': sentences2[j],
                        'sim_score': sim
                    }
                    plagiarism_matches.append(match_info)
        
        # calculate metrics
        if count_aligned > 0:
            average_similarity = total_similarity / count_aligned
        else:
            average_similarity = 0
        
        max_size = len(sentences1)
        if len(sentences2) > max_size:
            max_size = len(sentences2)
        
        if max_size > 0:
            plagiarism_ratio = len(plagiarism_matches) / max_size
        else:
            plagiarism_ratio = 0
        
        # determine if plagiarized
        is_plagiarized = False
        if plagiarism_ratio > 0.5 or average_similarity > self.thresh:
            is_plagiarized = True
        
        final_results = {
            'matches': plagiarism_matches,
            'match_count': len(plagiarism_matches),
            'total_pairs': len(alignment_result),
            'avg_sim': average_similarity,
            'plag_ratio': plagiarism_ratio,
            'plagiarized': is_plagiarized,
            'expanded': self.nodes_checked
        }
        
        return final_results
    
    # display the results
    def print_results(self, results):        
        if results['plagiarized']:
            print("\nResult: PLAGIARISM DETECTED")
        else:
            print("\nResult: No plagiarism found")
        
        print(f"\nPlagiarism ratio: {results['plag_ratio']*100:.1f}%")
        print(f"Avg similarity: {results['avg_sim']*100:.1f}%")
        print(f"Similar pairs: {results['match_count']}/{results['total_pairs']}")
        print(f"Nodes expanded: {results['expanded']}")
        
        # show some matches
        if len(results['matches']) > 0:
            print(f"\nMatching sentences found:")
            counter = 0
            for match in results['matches']:
                if counter >= 3:
                    break
                counter = counter + 1
                print(f"\n{counter}. Similarity: {match['sim_score']*100:.1f}%")
                print(f"   Doc1: {match['sentence1'][:70]}...")
                print(f"   Doc2: {match['sentence2'][:70]}...")
        

def test_system():
    checker = PlagiarismChecker(thresh=0.7)
    
    # test with identical docs
    print("\nTest 1: Identical documents")
    txt1 = "We live in Gandhinagar. Our college is IIIT Vadodara. The campus has a peaceful environment."
    txt2 = "We live in Gandhinagar. Our college is IIIT Vadodara. The campus has a peaceful environment."
    result = checker.check_plagiarism(txt1, txt2)
    checker.print_results(result)
    
    # test with modified text
    print("\nTest 2: Modified document")
    txt1 = "We live in Gandhinagar. Our college is IIIT Vadodara. The campus has a peaceful environment."
    txt2 = "Our residence is in Gandhinagar city. We study at IIIT Vadodara. The surroundings of the campus are calm and pleasant."
    result = checker.check_plagiarism(txt1, txt2)
    checker.print_results(result)
    
    # test completely different
    print("\nTest 3: Different documents")
    txt1 = "We live in Gandhinagar. Our college is IIIT Vadodara."
    txt2 = "The weather in Shimla is cold. I enjoy trekking in the mountains. Winter brings snowfall and hot coffee."
    result = checker.check_plagiarism(txt1, txt2)
    checker.print_results(result)
    
    # test partial overlap
    print("\nTest 4: Partial overlap")
    txt1 = "We live in Gandhinagar. Our college is IIIT Vadodara. The professors are good."
    txt2 = "Our college is IIIT Vadodara. We attended a tech fest last week. The professors are good."
    result = checker.check_plagiarism(txt1, txt2)
    checker.print_results(result)


if __name__ == "__main__":
    test_system()
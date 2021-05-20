import random
'''
Selection Criteria
    - Max: Select the entry as final output which occurs the most number of times
    - Roulette: Select the entry as final output using random selection by
                appropriately weighing the probability of each distinct entry 
                based on their number of occurences in the list 
'''
class selectFinal:

    def select_max(count):
        keymax = max(count, key = lambda x: count[x])
        return keymax

    def select_roulette(count):
        sum = 0
        for key in count:
            sum = sum + count[key]
        r = random.randint(0, sum)
        print(r)
        
        for key in count:        
            r = r - count[key]
            if r <= 0:
                keymax = key
                break
        
        return keymax
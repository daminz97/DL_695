import string
import random

random.seed(0)

# class People
class People:
    def __init__(self, f_arr, m_arr, l_arr, order):
        self.first_names = f_arr
        self.middle_names = m_arr
        self.last_names = l_arr
        self.order = order
    
    def __iter__(self):
        return PeopleIterator(self)
    
    def __call__(self):
        for name in sorted(self.last_names):
            print(name)
        print()
        return
    
class PeopleIterator:
    def __init__(self, people_obj):
        self.f_names = people_obj.first_names
        self.m_names = people_obj.middle_names
        self.l_names = people_obj.last_names
        self.order = people_obj.order
        self.index = -1
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self.index += 1
        if self.index < len(self.f_names):
            if self.order == 'first_name_first':
                return ' '.join((self.f_names[self.index], self.m_names[self.index], self.l_names[self.index]))
            if self.order == 'last_name_first':
                return ' '.join((self.l_names[self.index], self.f_names[self.index], self.m_names[self.index]))
            if self.order == 'last_name_with_comma_first':
                return ' '.join((self.l_names[self.index] + ',', self.f_names[self.index], self.m_names[self.index]))
        else:
            print()
            raise StopIteration
            
    next = __next__
# end of class People definition

# class PeopleWithMoney
class PeopleWithMoney(People):
    def __init__(self, f_arr, m_arr, l_arr, order, wealth):
        super(PeopleWithMoney, self).__init__(f_arr, m_arr, l_arr, order)
        self.wealth = wealth
        
    def __iter__(self):
        return PWMIterator(self)
    
    def __call__(self):
        concat_list = sorted(zip(self.first_names, self.middle_names, self.last_names, self.wealth), key=lambda pair: pair[3])
        for f, m, l, w in concat_list:
            print(' '.join((f, m, l, str(w))))
        return

class PWMIterator(PeopleIterator):
    def __init__(self, pwm_obj):
        super(PWMIterator, self).__init__(pwm_obj)
        self.wealths = pwm_obj.wealth
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.index += 1
        if self.index < len(self.f_names):
            return ' '.join((self.f_names[self.index], self.m_names[self.index], self.l_names[self.index], str(self.wealths[self.index])))
        else:
            print()
            raise StopIteration

    next = __next__
# end of class PeopleWithMoney definition

# test code
def main():
    def name_generator():
        arr = []
        for i in range(10):
            name = ''.join(random.choice(string.ascii_lowercase) for j in range(5))
            arr.append(name)
        return arr

    f_names = name_generator()
    m_names = name_generator()
    l_names = name_generator()
    money = [random.randint(0,1000) for i in range(10)]
    formats = ['first_name_first', 'last_name_first', 'last_name_with_comma_first']

    for item in formats:
        real_people = People(f_names, m_names, l_names, item)
        iters = iter(real_people)
        
        # 5-6
        for people in iters:
            print(people)

    real_people = People(f_names, m_names, l_names, 'first_name_first')
    real_people()

    # 7
    people_money = PeopleWithMoney(f_names, m_names, l_names, 'first_name_first', money)
    iters2 = iter(people_money)

    for people in iters2:
        print(people)
    
    people_money()

main()
# end of test code
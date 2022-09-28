
from Filters import FILTERS
from FilterLearning import Agent
import pickle

ozone = 5  #This value represents the ozone intensity that the sensor will detect.
QL = Agent(ozone)
q_file = QL.check_q()
with open('brain', 'rb') as q_file:
  q_table=pickle.load(q_file)
if q_table is not None:
  if ozone in q_table.keys():
    filter = FILTERS(ozone, 7)
    current_filter = 0
    q_t = q_table[ozone]
    action = q_t.loc[0, :].idxmax()
    while current_filter != filter.filtersno-1:
      next_filter, R = filter.next_states(current_filter, action)
      current_filter = next_filter        
      action_ = q_t.loc[next_filter, :].idxmax()
      action = action_
    print("Ozone levels of filters: ", filter.states)
    print("\nQ-table:\n", q_t)
  else:
    QL.rl()
else:
  QL.rl()

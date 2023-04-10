import csv
from CartPole.CPstates import Grid
from CartPole.CPlearn import init_state_transition_count

# write job parameters
def write_parameters(interval,begin,end,filename):
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([interval,begin,end])
    csvfile.close()

#read job parameters
def read_parameters(filename):
    with open(filename,'r') as csvfile:
        csvreader = csv.reader(csvfile)
        param_info = csvreader.__next__()
        interval_str = param_info[0]
        begin_str = param_info[1]
        end_str = param_info[2]
    return interval_str,begin_str,end_str

# write grid parameters to a csv file
def write_grid(grid,filename):
    with open(filename,'w') as csvfile:
        gridwriter = csv.writer(csvfile)
        a = grid.obs_interval[0][1]
        b = grid.obs_interval[1][1]
        c = grid.obs_interval[2][1]
        d = grid.obs_interval[3][1]
        nbins = grid.bins[0]                
        gridwriter.writerow([nbins,a,b,c,d])
    csvfile.close()

# read grid parameters from a csv file
def read_grid(filename):
    with open(filename,'r') as csvfile:
        gridreader = csv.reader(csvfile)
        grid_info = gridreader.__next__()
        nbins = int(grid_info[0])
        a = float(grid_info[1])
        b = float(grid_info[2])
        c = float(grid_info[3])
        d = float(grid_info[4])
        grid = Grid(nbins,a,b,c,d)
    csvfile.close()
    return grid

# write Q values to a csv file
def write_Q(Q,filename):
    with open(filename,'w') as csvfile:
        Qwriter = csv.writer(csvfile)
        for state in Q:
            Qwriter.writerow([state[0], state[1], state[2], state[3], Q[state][0], Q[state][1]])
    csvfile.close()

# reads Q values from a csv file
def read_Q(filename):
    with open(filename, 'r') as csvfile:
        Qreader = csv.reader(csvfile)
        Q = {}
        for row in Qreader:
            w = int(row[0])
            x = int(row[1])
            y = int(row[2])
            z = int(row[3])
            state = (w,x,y,z)
            Q[state] = {}
            Q[state][0] = float(row[4])
            Q[state][1] = float(row[5])
    csvfile.close()
    return Q

# writes game_list to csv file
def write_game_length(list,filename):
    with open(filename,'w') as csvfile:
        writer = csv.writer(csvfile)
        for item in list:
            writer.writerow([item])
    csvfile.close()

# reads game_list from csv file
def read_game_length(filename):
    list = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            item = int(row[0])
            list.append(item)
    csvfile.close()
    return list

# writes norm to a csv file
def write_norm(list,filename):
    with open(filename,'w') as csvfile:
        writer = csv.writer(csvfile)
        for item in list:
            writer.writerow([item])
    csvfile.close()

# reads norm from a csvfile
def read_norm(filename):
    list = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            item = float(row[0])
            list.append(item)
    csvfile.close()
    return list

# writes terminal states to csv file
def write_terminal_states(list,filename):
    with open(filename,'w') as csvfile:
        writer = csv.writer(csvfile)
        for item in list:
            writer.writerow(item)
    csvfile.close()

# reads terminal states from a csv file
def read_terminal_states(filename):
    list = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            a = int(row[0])
            b = int(row[1])
            c = int(row[2])
            d = int(row[3])
            list.append((a,b,c,d))
    csvfile.close()
    return list

# writes a disctionary of states and values to a csvfile
def write_states(states,filename):
    with open(filename,'w') as csvfile:
        statewriter = csv.writer(csvfile)
        for state in states:
            statewriter.writerow([state[0], state[1], state[2], state[3], states[state]])
    csvfile.close()

# reads a dictionary of states and values from a csvfile
def read_states(filename):
    with open(filename, 'r') as csvfile:
        statereader = csv.reader(csvfile)
        states = {}
        for row in statereader:
            w = int(row[0])
            x = int(row[1])
            y = int(row[2])
            z = int(row[3])
            state = (w,x,y,z)
            states[state] = int(row[4])
    csvfile.close()
    return states
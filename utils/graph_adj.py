import numpy
import torch
import pandas
import scipy.sparse as sp

def get_graph_adj(args):
    file = args.data_dir
    file_save = args.save_dir
    file_label = file + "/train/label"
    file_seq_out = file + "/train/seq.out"
    label = []
    seq_out = []
    file_dict_intent = file_save + "/alphabet/intent_dict.txt"
    file_dict_slot = file_save + "/alphabet/slot_dict.txt"
    intent_dict = {}
    slot_dict = {}
    num_intent = 0
    num_slot = 0
    with open(file_dict_intent) as i, open(file_dict_slot) as s:
        for line in i:
            num_intent += 1
        for line in s:
            num_slot += 1
    
    stat = [[0] * num_slot for _ in range(num_intent)]
    intent_stat = [0] * num_intent
    slot_stat = [0] * num_slot
    with open(file_label) as r1, open(file_seq_out) as r3:
        for l1 in r1:
            l3 = r3.readline()
            label.append(l1.strip())
            seq_out.append(l3.strip())
    with open(file_dict_intent) as i, open(file_dict_slot) as s:
        for line in i:
            line = line.split('\t')
            intent_dict[line[0]] = line[1]
        for line in s:
            line = line.split('\t')
            slot_dict[line[0]] = line[1]
    for la, so in zip(label, seq_out):
        if '#' not in la:
            intent_id = int(intent_dict[la])
            intent_stat[intent_id] += 1
            slot_list = so.split(' ')
            for slot in slot_list:
                if slot in slot_dict.keys():
                    slot_id = int(slot_dict[slot])
                    slot_stat[slot_id] += 1
                    stat[intent_id][slot_id] += 1
        elif '#' in la:
            intent_list = la.split('#')
            for i in intent_list:
                intent_id = int(intent_dict[i])
                intent_stat[intent_id] += 1
                slot_list = so.split(' ')
                for slot in slot_list:
                    if slot in slot_dict.keys():
                        slot_id = int(slot_dict[slot])
                        slot_stat[slot_id] += 1
                        stat[intent_id][slot_id] += 1
        else:
            print('error!!!')

    data = numpy.array(stat)
    data_intent = numpy.array(intent_stat)
    data_slot = numpy.array(slot_stat)
    data = numpy.delete(data, [0], axis=1)
    data_slot = numpy.delete(data_slot, [0])

    o_1 = numpy.array([0])
    o_2 = numpy.array([[0] * num_intent])
    data_slot = numpy.insert(data_slot, 0, o_1)
    data = numpy.insert(data, 0, o_2, axis=1)

    numpy.savetxt("./save/data_intent.txt", data_intent)
    numpy.savetxt("./save/data_slot.txt", data_slot)
    numpy.savetxt("./save/data.txt", data)

    df = pandas.DataFrame(data)
    df_intent = pandas.Series(data_intent)
    df_slot = pandas.Series(data_slot)
    intent_2_slot = df.div(df_intent + 1e-10, axis=0)
    slot_2_intent = df.div(df_slot + 1e-10, axis=1)
    
    A_right_up = intent_2_slot.values
    A_left_down = slot_2_intent.values.T
    A_left_up = numpy.array([[0] * num_intent for _ in range(num_intent)])
    A_right_down = numpy.array([[0] * (num_slot) for _ in range(num_slot)])
    A_left = numpy.concatenate((A_left_up, A_left_down), axis=0)
    A_right = numpy.concatenate((A_right_up, A_right_down), axis=0)
    A = numpy.concatenate((A_left, A_right), axis=1)
    
    A_eye = numpy.eye(num_intent + num_slot)
    A = A + A_eye
    numpy.savetxt("./save/graph_adj.txt", A)
    adj = torch.from_numpy(A).float()
    adj = sp.coo_matrix(adj)
    rowsum = numpy.array(adj.sum(1))
    d_inv_sqrt = numpy.power(rowsum, -0.5).flatten()
    d_inv_sqrt[numpy.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    output_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    output_adj = torch.FloatTensor(numpy.array(output_adj.todense()))
    numpy.savetxt("./save/graph_output_adj.txt", numpy.array(output_adj))

    return output_adj


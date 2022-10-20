import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils import graph_adj


class QKVAttention(nn.Module):
    
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-1, -2)
        ) / math.sqrt(self.__hidden_dim), dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class EnSelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(EnSelfAttention, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        return attention_x


class Attention(nn.Module):
    def __init__(self, dropout_rate):
        super(Attention, self).__init__()
    
    def forward(self, input_query, input_key, input_value):
        score_tensor = F.softmax(torch.matmul(
            input_query,
            input_key.transpose(-2, -1)
        ), dim=-1)
        forced_tensor = torch.matmul(score_tensor, input_value)
        return forced_tensor 

class GraphConvolution(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hw_linear = nn.Linear(input_dim, output_dim)

    def forward(self, input_feature, adjacency):
        ah = torch.matmul(adjacency, input_feature)
        output = self.hw_linear(ah)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.input_dim) + '->' + str(self.output_dim) + ')'


class GcnNet(nn.Module):
    def __init__(self, args, input_dim, output_dim, dropout_rate):
        super(GcnNet, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.conv1 = GraphConvolution(self.args, self.input_dim, self.output_dim)

    def forward(self, node_features, adjacency):
        node_features = self.conv1(node_features, adjacency)
        node_features = F.dropout(node_features, self.dropout_rate, training=self.training)
        output = node_features
        return output


class BiLSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(BiLSTMEncoder, self).__init__()
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        dropout_text = self.__dropout_layer(embedded_text)
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)
        return padded_hiddens


class TaskSharedEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.__args = args

        self.__attention = EnSelfAttention(
            self.__args.word_embedding_dim,
            self.__args.self_attention_hidden_dim,
            self.__args.self_attention_output_dim,
            self.__args.dropout_rate
        )

        self.__encoder = BiLSTMEncoder(
            self.__args.word_embedding_dim,
            self.__args.encoder_hidden_dim,
            self.__args.dropout_rate
        )

    def forward(self, word_tensor, seq_lens):
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=2)
        return hiddens


class IntraUtteranceAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(IntraUtteranceAttention, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__w1_layer = nn.Linear(self.__input_dim, self.__hidden_dim, bias=False)
        self.__w2_layer = nn.Linear(self.__hidden_dim, self.__output_dim, bias=False)

    def forward(self, input_x, seq_lens):
        input_x = self.__dropout_layer(input_x)
        o_w1 = self.__w1_layer(input_x)
        o_w1 = F.tanh(o_w1) 
        o_w2 = self.__w2_layer(o_w1)
        value = F.softmax(o_w2, dim=-1)
        attention_x = torch.matmul(value.transpose(-1, -2), input_x)

        return attention_x


class SlotDecoderBlock(nn.Module):
    def __init__(self, args, hidden_size):
        super(SlotDecoderBlock, self).__init__()
        self.__args = args

        self.dense_in = nn.Linear(hidden_size * 6, hidden_size)
        self.act_fun = nn.ReLU()
        self.dense_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.__args.dropout_rate)

    def forward(self, intent_tensor, slot_tensor):
        cat_tensor = torch.cat([intent_tensor, slot_tensor], dim=2)
        batch_size, max_length, hidden_size = cat_tensor.size()
        h_pad = torch.zeros(batch_size, 1, hidden_size)
        if self.__args.gpu and torch.cuda.is_available():
            h_pad = h_pad.cuda()
        h_left = torch.cat([h_pad, cat_tensor[:, :max_length - 1, :]], dim=1)
        h_right = torch.cat([cat_tensor[:, 1:, :], h_pad], dim=1)
        cat_tensor = torch.cat([cat_tensor, h_left, h_right], dim=2)

        cat_tensor = self.dense_in(cat_tensor)
        cat_tensor = self.act_fun(cat_tensor)
        cat_tensor =self.dense_out(cat_tensor)
        cat_tensor = self.dropout(cat_tensor)
        slot_tensor_new = cat_tensor + slot_tensor
        return slot_tensor_new


class ModelManager(nn.Module):
    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        # word embedding
        self.__embedding = nn.Embedding(self.__num_word, self.__args.word_embedding_dim)
        # task-shared: self-attentive encoder
        self.__text_encoder = TaskSharedEncoder(args)
        # task-specific encoder
        self.__text_lstm_intent = BiLSTMEncoder(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
                                                 self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
                                                 self.__args.dropout_rate)
        self.__text_lstm_slot = BiLSTMEncoder(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
                                                 self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
                                                 self.__args.dropout_rate) 

        # intra-corpus label embedding is updated during training
        # intent label embedding
        self.__intent_embedding = nn.Parameter(torch.FloatTensor(self.__num_intent, self.__args.intent_embedding_dim))
        nn.init.normal_(self.__intent_embedding.data)
        # slot label embedding
        self.__slot_embedding = nn.Parameter(torch.FloatTensor(self.__num_slot, self.__args.slot_embedding_dim))
        nn.init.normal_(self.__slot_embedding.data)
        
        # intra-utterance attention for intent and slot label representation
        self.__intent_attention = IntraUtteranceAttention(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
                                               self.__args.self_attention_hidden_dim,
                                               self.__num_intent,
                                               self.__args.dropout_rate)
        self.__slot_attention = IntraUtteranceAttention(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
                                               self.__args.self_attention_hidden_dim,
                                               self.__num_slot,
                                               self.__args.dropout_rate)

        # intent-slot co-occurrence gcn
        self.graph_Adj = graph_adj.get_graph_adj(self.__args)
        if torch.cuda.is_available():
            self.graph_Adj = self.graph_Adj.cuda()
        self.graph_Adj.requires_grad = False
        self.__graph_gcn = GcnNet(self.__args,
                                  self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim,
                                  self.__args.gcn_output_dim,
                                  self.__args.gcn_dropout_rate)

        # intent and slot embedding adaptively fuse with w1 and w2
        self.__intent_weight1 = nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, 1)
        self.__intent_weight2 = nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, 1)
        self.__slot_weight1 = nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, 1)
        self.__slot_weight2 = nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, 1)
        
        # extract intent and slot information from utterance using in slot decoder
        self.__intent_text_qkv = Attention(self.__args.dropout_rate)
        self.__slot_text_qkv = Attention(self.__args.dropout_rate)
        # information fuse block used in slot decoder
        self.__fuse_block = SlotDecoderBlock(self.__args, self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim)

        # intent decoder mlp
        self.__intent_decoder = nn.Sequential(
            nn.Linear(self.__num_intent, self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, self.__num_intent)
        )
        
        # slot decoder mlp
        self.__slot_decoder = nn.Sequential(
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim),
            nn.Dropout(self.__args.slot_decoder_dropout_rate),
            nn.LeakyReLU(args.alpha),
            nn.Linear(self.__args.encoder_hidden_dim + self.__args.self_attention_output_dim, self.__num_slot)
        )


    def forward(self, text, seq_lens, n_predicts=None):
        word_tensor = self.__embedding(text)

        # utterance encoder
        # task-shared
        text_encoder = self.__text_encoder(word_tensor, seq_lens)
        # task-specific
        text_hiddens_intent = self.__text_lstm_intent(text_encoder, seq_lens)
        text_hiddens_intent = F.dropout(text_hiddens_intent, p=self.__args.dropout_rate, training=self.training)
        text_hiddens_slot = self.__text_lstm_slot(text_encoder, seq_lens)
        text_hiddens_slot = F.dropout(text_hiddens_slot, p=self.__args.dropout_rate, training=self.training)

        seq_lens_tensor = torch.tensor(seq_lens)
        if self.__args.gpu:
            seq_lens_tensor = seq_lens_tensor.cuda()

        # intent and slot label embedder
        # intra-utterance
        intent_attention_out = self.__intent_attention(text_hiddens_intent, seq_lens)
        intent_attention_out = F.dropout(intent_attention_out, p=self.__args.dropout_rate, training=self.training)
        slot_attention_out = self.__slot_attention(text_hiddens_slot, seq_lens)
        slot_attention_out = F.dropout(slot_attention_out, p=self.__args.dropout_rate, training=self.training)
        # intra-corpus
        intent_embedding = torch.matmul(torch.matmul(self.__intent_embedding.unsqueeze(0).repeat(len(seq_lens), 1, 1), text_hiddens_intent.transpose(-1, -2)), text_hiddens_intent)
        slot_embedding = torch.matmul(torch.matmul(self.__slot_embedding.unsqueeze(0).repeat(len(seq_lens), 1, 1), text_hiddens_slot.transpose(-1, -2)), text_hiddens_slot)
        # adaptive fusion
        intent_weight1 = torch.sigmoid(self.__intent_weight1(intent_embedding))
        intent_weight2 = torch.sigmoid(self.__intent_weight2(intent_attention_out))
        intent_weight1 = intent_weight1 / (intent_weight1 + intent_weight2)
        intent_weight2 = 1 - intent_weight1
        slot_weight1 = torch.sigmoid(self.__slot_weight1(slot_embedding))
        slot_weight2 = torch.sigmoid(self.__slot_weight2(slot_attention_out))
        slot_weight1 = slot_weight1 / (slot_weight1 + slot_weight2)
        slot_weight2 = 1 - slot_weight1
        intent_attention_out = intent_weight1 * intent_embedding + intent_weight2 * intent_attention_out
        intent_attention_out = F.dropout(intent_attention_out, p=self.__args.dropout_rate, training=self.training)
        slot_attention_out = slot_weight1 * slot_embedding + slot_weight2 * slot_attention_out
        slot_attention_out = F.dropout(slot_attention_out, p=self.__args.dropout_rate, training=self.training)

        # intent-slot co-occurrence gcn
        graph_H = torch.cat([intent_attention_out, slot_attention_out], dim=1)
        graph_Adj = self.graph_Adj.unsqueeze(0).repeat(len(seq_lens), 1, 1)
        graph_H = self.__graph_gcn(graph_H, graph_Adj) 
        # updated label representation through gcn
        intent_label_H = graph_H[:, 0:self.__num_intent, :]
        slot_label_H = graph_H[:, self.__num_intent:, :]

        # similarity of intent and utterance  
        text_fuse_intent_pred = torch.matmul(text_hiddens_intent, intent_label_H.transpose(-1, -2))

        # information enhance and fuse block in slot decoder
        text_fuse_intent = self.__intent_text_qkv(text_hiddens_intent, intent_label_H, intent_label_H)
        text_fuse_intent = text_fuse_intent + text_hiddens_intent
        text_fuse_slot = self.__slot_text_qkv(text_hiddens_slot, slot_label_H, slot_label_H)
        text_fuse_slot = text_fuse_slot + text_hiddens_slot
        text_fuse_slot = self.__fuse_block(text_fuse_intent, text_fuse_slot)

        #intent decoder mlp
        pred_intent = self.__intent_decoder(text_fuse_intent_pred)
        
        #slot decoder mlp
        pred_slot = self.__slot_decoder(text_fuse_slot)

        pred_slot_list = []
        for i in range(0, len(seq_lens)):
            pred_slot_list.append(pred_slot[i, 0:seq_lens[i], :])
        pred_slot = torch.cat(pred_slot_list, dim=0)
        pred_slot = F.log_softmax(pred_slot, dim=1)

        if n_predicts is None:
            return pred_slot, pred_intent
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)

            intent_index_sum = torch.cat(
                [
                    torch.sum(torch.sigmoid(pred_intent[i, 0:seq_lens[i], :]) > self.__args.threshold, dim=0).unsqueeze(
                        0)
                    for i in range(len(seq_lens))
                ],
                dim=0
            )
            intent_index = (intent_index_sum > (seq_lens_tensor // 2).unsqueeze(1)).nonzero()

            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))
        print('\tword embedding dimension:				    {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				    {};'.format(self.__args.encoder_hidden_dim))
        print('\tdimension of intent embedding:		    	{};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot embedding:		    	{};'.format(self.__args.slot_embedding_dim))
        print('\toutput dimension of gcn graph:             {};'.format(self.__args.gcn_output_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

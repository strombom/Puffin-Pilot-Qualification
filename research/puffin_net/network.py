
import random
import numpy as np


class Neuron:
    dendrite_count = 5
    synapse_count = 20
    synapse_threshold = 10
    max_permanence = 50

    def __init__(self):
        # Synapse: row_idx, col_idx, permanence
        self.synapses = np.zeros((self.dendrite_count, self.synapse_count, 3), dtype = np.int32)
        self.inputs = []

    def append_inputs(self, inputs):
        self.inputs.append(inputs)

    def input_index(self, input_idx):
        row_idx, col_idx = 0, 0
        for input_row in self.inputs:
            if input_idx < input_row.shape[0]:
                col_idx = input_idx
                break
            else:
                input_idx -= input_row.shape[0]
            row_idx += 1
        return row_idx, col_idx

    def init_synapses(self):
        input_count = 0
        for input_row in self.inputs:
            input_count += input_row.shape[0]

        for dendrite in self.synapses:
            for synapse in dendrite:
                input_idx = random.randint(0, input_count-1)
                row_idx, col_idx = self.input_index(input_idx)
                synapse[0], synapse[1] = row_idx, col_idx
                synapse[2] = random.randint(0, self.max_permanence)


class MiniColumn:
    neuron_count = 16

    def __init__(self, output):
        self.output = output
        self.neurons = []
        for neuron_idx in range(self.neuron_count):
            neuron = Neuron()
            self.neurons.append(neuron)

    def append_inputs(self, inputs):
        for neuron in self.neurons:
            neuron.append_inputs(inputs)

    def init_synapses(self):
        for neuron in self.neurons:
            neuron.init_synapses()


class Column:
    minicolumn_count = 8 * 2

    def __init__(self, outputs):
        self.bot_outputs = outputs[0:self.minicolumn_count//2]
        self.top_outputs = outputs[self.minicolumn_count//2:]

        self.bot_minicolumns = []
        self.top_minicolumns = []
        for minicolumn_idx in range(self.minicolumn_count):
            minicolumn = MiniColumn(outputs[minicolumn_idx:minicolumn_idx+1])
            if minicolumn_idx < self.minicolumn_count // 2:
                self.bot_minicolumns.append(minicolumn)
            else:
                self.top_minicolumns.append(minicolumn)

    def append_inputs(self, inputs):
        # Connect bottom half of minicolumns
        for minicolumn in self.bot_minicolumns:
            minicolumn.append_inputs(inputs)

    def append_context(self, inputs):
        # Connect top half of minicolumns
        for minicolumn in self.top_minicolumns:
            minicolumn.append_inputs(inputs)

    def get_outputs(self):
        # Return outputs for top half of minicolumns
        return self.top_outputs

    def init_columns(self):
        # Connect bottom and top half of the minicolumns
        for minicolumn in self.top_minicolumns + self.bot_minicolumns:
            minicolumn.append_inputs(self.bot_outputs)
            minicolumn.append_inputs(self.top_outputs)

        for minicolumn in self.top_minicolumns + self.bot_minicolumns:
            minicolumn.init_synapses()

class Region:
    column_count = 3

    def __init__(self, outputs):
        self.columns = []
        for column_idx in range(self.column_count):
            column = Column(outputs[column_idx])
            self.columns.append(column)
    
    def append_inputs(self, inputs, stride, overlap):
        assert overlap <= self.column_count

        input_group_size = inputs[0].shape[0]
        input_group_count = len(inputs)
        input_total_count = len(inputs) * input_group_size

        input_width = input_total_count / self.column_count
        view_width_half = input_width * overlap * 0.5

        for idx, column in enumerate(self.columns):
            center = input_width * (idx + 0.5)
            start_idx = int(round(center - view_width_half))
            stop_idx = int(round(center + view_width_half))
            if start_idx < 0:
                start_idx = 0
            if stop_idx >= input_total_count:
                stop_idx = input_total_count - 1


            start_group_idx = start_idx // input_group_size
            stop_group_idx = stop_idx // input_group_size
            
            #print("--")
            for group_idx in range(start_group_idx, stop_group_idx+1):
                if group_idx == start_group_idx:
                    start_item_idx = start_idx % input_group_size
                else:
                    start_item_idx = 0

                if group_idx == stop_group_idx:
                    stop_item_idx = stop_idx % input_group_size
                else:
                    stop_item_idx = input_group_size - 1

                column_inputs = inputs[group_idx][start_item_idx:stop_item_idx+1:stride]
                column.append_inputs(column_inputs)
                #print("Group", group_idx, start_item_idx, stop_item_idx, column_inputs)


    def get_outputs(self):
        outputs = []
        for column in self.columns:
            outputs.append(column.get_outputs())
        return outputs

    def init_columns(self):
        # Connect context between top columns
        for idx in range(self.column_count - 1):
            outputs = self.columns[idx].top_outputs
            self.columns[idx + 1].append_context(outputs)
            outputs = self.columns[self.column_count - idx - 1].top_outputs
            self.columns[self.column_count - idx - 2].append_context(outputs)

        # Init synapses
        for column in self.columns:
            column.init_columns()


class Network:
    input_count = 16
    region_count = 2

    def __init__(self):
        self.inputs = np.zeros((self.input_count))
        self.minicolumn_outputs = np.zeros((self.region_count, 
                                            Region.column_count,
                                            Column.minicolumn_count),
                                           dtype = np.int8)

        self.regions = []
        for region_idx in range(self.region_count):
            region = Region(self.minicolumn_outputs[region_idx])
            self.regions.append(region)
        
        self.regions[0].append_inputs([self.inputs], stride = 1, overlap = 1.5)
        self.regions[1].append_inputs([self.inputs], stride = 2, overlap = 2.0)
        #self.regions[2].append_inputs([self.inputs], stride = 3, overlap = 2.5)

        self.regions[1].append_inputs(self.regions[0].get_outputs(), stride = 1, overlap = 1.5)
        #self.regions[2].append_inputs(self.regions[1].get_outputs(), stride = 1, overlap = 1.5)
        #self.regions[2].append_inputs(self.regions[0].get_outputs(), stride = 2, overlap = 2.0)

        for region_idx in range(self.region_count):
            self.regions[region_idx].init_columns()

    def tick(self, input_data):
        pass


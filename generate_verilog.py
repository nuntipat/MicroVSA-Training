import numpy as np
import logicmin
import math
import os
import argparse

#################################
#
# utility functions and constant
#
#################################

def binary_to_bipolar(v):
    v[v == 1] = -1
    v[v == 0] = 1
    return v

def bipolar_to_binary(v):
    v[v == 1] = 0
    v[v == -1] = 1
    return v

lutn_input_binary = [
    np.array([[]], dtype=int)
]
lutn_input_bipolar = []
lutn_input_str = []
sum_results = []
sum_sols = []

def init_constant(lut_size):
    global lutn_input_bipolar, lutn_input_str

    for i in range(lut_size):
        lutn_input_binary.append(np.array([list(np.binary_repr(n, i+1)) for n in range(pow(2, i+1))], dtype=int),)

    lutn_input_bipolar = [binary_to_bipolar(v.copy()) for v in lutn_input_binary]
    lutn_input_str = [[''.join(row) for row in v.astype(str).tolist()] for v in lutn_input_binary]

    for num_vars, arr in enumerate(lutn_input_bipolar):
        tmp = np.sum(arr, axis=1) 
        tmp[tmp >= 0] = 0
        tmp[tmp < 0] = 1

        t = logicmin.TT(num_vars, 1)
        for i, o in zip(lutn_input_str[num_vars], tmp):
            t.add(i, str(o))
        sols = t.solve()

        sum_results.append(tmp)
        sum_sols.append(sols)

########################################
#
# helper functions for code generation
#
########################################

def generate_input_name(level, col, group_id, var_index):
    if level == 0:
        return [f'Vf[{n}][{col}]' for n in var_index]
    else:
        return [f'lv{level-1}[{n}][{col}]' for n in var_index]
    
def generate_output_name(level, col, group_id, var_index):
    if level == 0:
        return f'lv0[{group_id}][{col}]'
    else:
        return f'lv{level}[{group_id}][{col}]'

# for 3-inputs lut x[0]-msb x[5]-lsb
# {
#   inputs: [0, 3, 5], 
#   output: [1, 0, 0, 0, 1, 0, 1, 1]} 
# }
def generate_lut(level, weight, mask, num_pad_row=0):
    num_row, _ = weight.shape
    if mask is not None:
        mask_pad = np.pad(mask, ((0, num_row - mask.shape[0]), (0, 0)))

    num_group = num_row//lut_size
    luts = []
    for i in range(fhv_dimension):
        for j in range(0, num_group):
            w = weight[(j*lut_size):(j*lut_size)+lut_size, i]
            if mask is not None:
                w = w * mask_pad[(j*lut_size):(j*lut_size)+lut_size, i]
            if j == num_group - 1:
                w[-num_pad_row:] = 0
            input_index = np.flatnonzero(w) + (j * lut_size)
            if input_index.shape[0] != 0:
                # TODO: validate that sum is not 0 before sign()
                s = bipolar_to_binary(np.sign(np.sum(lutn_input_bipolar[input_index.shape[0]] * w[w != 0], axis=1)).astype(int))
                lut = {'inputs': input_index, 'output': np.sign(s), 
                       'input_names': generate_input_name(level, i, j, input_index),
                       'output_name': generate_output_name(level, i, j, input_index)}
            # else:
            #     lut = {'inputs': np.array([], dtype=int), 'output': np.array([], dtype=int),
            #            'input_names': [],
            #            'output_name': generate_output_name(level, i, j, input_index)}
                luts.append(lut) 
    return luts

def getUsedVariableMask(sol):
    if len(sol.sols[0].cubes) == 0: # y => 0  no cube, y => 1 t = 0 and f = 0
        return '0'
    variableUsed = np.bitwise_or.reduce([c.t for c in sol.sols[0].cubes]) | np.bitwise_or.reduce([c.f for c in sol.sols[0].cubes])
    return np.binary_repr(variableUsed, sol.sols[0].X_MAX_VARS)

def getNumUsedVariable(sol):
    return getUsedVariableMask(sol).count('1')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate verilog code')
    parser.add_argument('-i', '--model-dir', help='path to the trained model (F, V, C, L, and P vector)')
    parser.add_argument('-o', '--output-dir', help='path to the dir to store the output verilog sources')
    parser.add_argument('-n', '--model-name', default='output', help='model name')
    parser.add_argument('-dv', '--v-dim', type=int, default=4, help='dimension of the V vector (Dv in the paper)')
    parser.add_argument('-ls', '--lut-size', type=int, default=6, help='number of LUT input')
    parser.add_argument('--enable-input-register', action='store_true', help='enable input register')
    parser.add_argument('--enable-add-register', action='store_true', help='enable add register')
    parser.add_argument('--add-register-stage', default='', help='comma separated list of stage id to add register')
    parser.add_argument('--enable-s-register', action='store_true', help='enable s register')
    parser.add_argument('--enable-p-register', action='store_true', help='enable p register')
    args = parser.parse_args()

    model_dir = args.model_dir
    model_name = args.model_name
    output_dir = args.output_dir
    vhv_dimension=args.v_dim
    lut_size=args.lut_size

    enable_input_register = args.enable_input_register
    enable_add_register = args.enable_add_register
    add_register_stage = [int(e) for e in args.add_register_stage.split(',')] if args.add_register_stage != '' else []
    enable_s_register = args.enable_s_register
    enable_p_register = args.enable_p_register

    init_constant(lut_size)

    # load model
    
    print ('Loading model...')

    F = np.load(f'{model_dir}/F.npy')
    V = np.load(f'{model_dir}/V.npy')
    C = np.load(f'{model_dir}/C.npy')
    P = np.load(f'{model_dir}/P.npy')
    L = np.load(f'{model_dir}/L.npz')
    L = [L[file] for file in L.files]
    M = np.ones_like(F, dtype=int)
    M[F==0] = 0

    num_input = F.shape[0]
    num_class = C.shape[0]
    num_value = V.shape[0]
    fhv_dimension = F.shape[1]
    class_prune_level = np.count_nonzero(C) / (num_class * fhv_dimension)

    print ()
    print ('Model Information:')
    print (f'Number of input = {num_input}')
    print (f'Number of class = {num_class}')
    print (f'Df = {fhv_dimension}')
    print (f'Number of non zero row in F = {np.count_nonzero(np.sum(F, axis=1))}')
    print (f'Percent of non zero weight = {np.count_nonzero(F) / (num_input * fhv_dimension)}')
    print (f'Percent of non zero class weight = {class_prune_level}')
    print ()

    # convert to lut

    print ('Generating LUT tree...')
    print ()

    luts = []
    opt_stats = []
    num_output_rows = num_input
    for layer_id, lut_weight in enumerate(L):
        print (f'Process layer {layer_id}...')
        num_lut_input, _ = lut_weight.shape
        if layer_id == 0:
            current_layer_mask = M
        else:
            current_layer_mask = np.zeros((num_lut_input, fhv_dimension))
            for j in range(fhv_dimension):
                for i in range(num_lut_input):
                    if f'lv{layer_id-1}[{i}][{j}]' in luts[layer_id-1]:
                        current_layer_mask[i][j] = 1
        current_level_luts = generate_lut(layer_id, lut_weight, current_layer_mask, num_lut_input - num_output_rows)
        
        # print (f'Pre-optimize utilization')
        # for n in range(7):    
        #     print(len([l for l in current_level_luts if l['inputs'].shape[0] == n]))

        opt_stat = np.zeros((lut_size, lut_size), dtype=int)
        for lut in current_level_luts:
            num_vars = lut['inputs'].shape[0]
            t = logicmin.TT(num_vars,1)
            for i, o in zip(lutn_input_str[num_vars], lut['output']):
                t.add(i, str(o))
                # print (i, o)
            sols = t.solve()
            opt_stat[num_vars-1][getNumUsedVariable(sols)-1] += 1
            lut['sol'] = sols
            lut['verilog'] = sols.printN(xnames=lut['input_names'],ynames=[lut['output_name']], syntax='Verilog')
        opt_stats.append(opt_stat)

        # print (f'Post-optimize utilization')
        print (opt_stat)

        tmp = {}
        for l in current_level_luts:
            tmp[l['output_name']] = l

        luts.append(tmp)
        num_output_rows = num_lut_input // lut_size
    
    last_layer_id = len(L)
    num_lut_input, _ = L[-1].shape

    tmp = {}
    for i in range(fhv_dimension):
        input_index = [j for j in range(num_lut_input // lut_size) if f'lv{last_layer_id-1}[{j}][{i}]' in luts[-1]]
        lut = {'inputs': np.array(input_index), 'output': sum_results[len(input_index)], 
                'input_names': generate_input_name(last_layer_id, i, 0, input_index),
                'output_name': f's[{i}]'}
        lut['verilog'] = sum_sols[len(input_index)].printN(xnames=lut['input_names'],ynames=[lut['output_name']], syntax='Verilog')
        tmp[lut['output_name']] = lut
        
    luts.append(tmp)

    # print utilization summary

    util_summary = np.add.reduce(opt_stats)
    print ('Logic minimization sumary')
    print (util_summary)
    print ('Estimate utilization for the encoding step')
    print (np.sum(util_summary, axis=0))
    print ()

    # Remove duplicate term in C to reduce number of adder

    print ('Optimizing similarity step...')

    num_remove = 0
    for j in range(fhv_dimension):
        tmp = C[0][j]
        should_remove = True
        for i in range(1, num_class):
            if C[i][j] != tmp:
                should_remove = False

        if should_remove and tmp != 0:
            for i in range(num_class):
                C[i][j] = 0
            num_remove += 1

    print (f'Removing: {num_remove} term(s) ({num_remove/fhv_dimension*100:.2f}%)')
    print ()

    # generate code

    used_vf = {}
    for lut in luts[0].values():
        for name in lut['input_names']:
            used_vf[name] = True 

    vf_to_v = {}
    for i in range(num_input):
        for j in range(fhv_dimension):
            if f'Vf[{i}][{j}]' in used_vf:
                if F[i][j] == -1:
                    vf_to_v[f'Vf[{i}][{j}]'] = f'(~V[{P[i]}][{j % vhv_dimension}])'
                else:
                    vf_to_v[f'Vf[{i}][{j}]'] = f'V[{P[i]}][{j % vhv_dimension}]'

    out_dir_path = f'{output_dir}/{model_name}'
    if enable_input_register:
        out_dir_path += 'v'
    if enable_add_register:
        out_dir_path += f"a{'-'.join([str(i) for i in add_register_stage])}" 
    if enable_s_register:
        out_dir_path += 's'
    if enable_p_register:
        out_dir_path += 'p'
    os.makedirs(f'{out_dir_path}', exist_ok=True)

    V_binary = bipolar_to_binary(V.copy().astype(int))
    with open(f'{out_dir_path}/v_mem.sv', 'w') as fp:
        print ('module v_mem (i, o);', file=fp)
        print (file=fp)
        print (f'input [7:0] i;', file=fp)
        print (f'output reg [0:{vhv_dimension-1}] o;', file=fp)
        print (file=fp)

        print ('always @(*)', file=fp)
        print ('case (i)', file=fp)
        for i in range(num_value):
            print (f"8'd{i}: o = 4'b{''.join(np.char.mod('%s', V_binary[i][:vhv_dimension]))};", file=fp)
        print ('endcase', file=fp)
        print (file=fp)

        print ('endmodule', file=fp)

    with open(f'{out_dir_path}/quantize.sv', 'w') as fp:
        print ('module vsalut_quantize (sample, V);', file=fp)
        print (file=fp)
        print (f'input [7:0] sample [0:{num_input-1}];', file=fp)
        print (f'output [0:{vhv_dimension-1}] V [0:{num_input-1}];', file=fp)
        print (file=fp)

        print ('genvar i;', file=fp)
        print ('generate', file=fp)
        print (f'for (i=0; i<{num_input}; i=i+1) begin : generate_vmem_module', file=fp)
        print (f'\tv_mem v_mem_inst (sample[i], V[i]);', file=fp)
        print ('end', file=fp)
        print ('endgenerate', file=fp)
        print (file=fp)

        print ('endmodule', file=fp)

    with open(f'{out_dir_path}/encode.sv', 'w') as fp:
        print ('module vsalut_encode (V, s);', file=fp)
        print (file=fp)
        print (f'input [0:{vhv_dimension-1}] V [0:{num_input-1}];', file=fp)
        print (f'output [0:{fhv_dimension-1}] s;', file=fp)
        print (file=fp)

        for i, lut_weight in enumerate(L[1:]):
            num_lut_input, _ = lut_weight.shape
            print (f'wire [0:{fhv_dimension-1}] lv{i} [0:{num_lut_input-1}];', file=fp)
            num_output_rows = num_lut_input // lut_size
        print (f'wire [0:{fhv_dimension-1}] lv{len(L)-1} [0:{num_output_rows-1}];', file=fp)
        print (file=fp)

        for level, lut_list in enumerate(luts):
            print (f'// lut level {level}', file=fp)
            for lut in lut_list.values():
                if level == 0:
                    code = lut['sol'].printN(xnames=[ vf_to_v[name] for name in lut['input_names'] ],ynames=[lut['output_name']], syntax='Verilog')
                else:
                    code = lut['verilog'].replace("= '0'", "= 1'b0")
                # print (f"assign {lut['verilog'].replace('<=', '=')};")
                print (f"assign {code.replace('<=', '=')};", file=fp)
            print (file=fp)

        print ('endmodule', file=fp)

    with open(f'{out_dir_path}/similarity.sv', 'w') as fp:
        target_output_bits = math.ceil(math.log2(fhv_dimension*class_prune_level))

        print ('module vsalut_similarity (', file=fp)
        if enable_add_register:
            print ('\tinput clk,', file=fp)
        print (f'\tinput [0:{fhv_dimension-1}] s,', file=fp)
        print (f'\toutput [{target_output_bits-1}:0] p [0:{num_class-1}]', file=fp)
        print (');', file=fp)
        print (file=fp)

        terms = []
        for i in range(num_class):
            term = []
            for j in range(fhv_dimension):
                if C[i][j] == 0:
                    pass
                elif C[i][j] == 1:
                    term.append(f's[{j}]')
                else:
                    term.append(f'~s[{j}]')
            terms.append(term)

        def create_always_block(current_stage):
            if enable_add_register and (current_stage in add_register_stage):
                print ('always @(posedge clk) begin', file=fp)
                return '<='
            else:
                print ('always @(*) begin', file=fp)
                return '='

        for i in range(num_class):
            # stage 0
            num_triple_stage0 = math.ceil(len(terms[i]) / 3)
            for j in range(num_triple_stage0):
                print (f'reg [1:0] class{i}_stage0_adder{j};', file=fp)
            print(file=fp)
            op = create_always_block(0)
            for j in range(num_triple_stage0):
                s = f'\tclass{i}_stage0_adder{j} {op} {terms[i][j*3]}'
                if j*3+1 < len(terms[i]):
                    s += f' + {terms[i][j*3+1]}'
                if j*3+2 < len(terms[i]):
                    s += f' + {terms[i][j*3+2]}'
                print (f'{s};', file=fp)
            print ('end', file=fp)
            print (file=fp)

            # stage 1
            num_triple_stage1 = math.ceil(num_triple_stage0 / 3)
            for j in range(num_triple_stage1):
                print (f'reg [3:0] class{i}_stage1_adder{j};', file=fp)
            print(file=fp)
            op = create_always_block(1)
            for j in range(num_triple_stage1):
                s = f'\tclass{i}_stage1_adder{j} {op} class{i}_stage0_adder{j*3}'
                if j*3+1 < num_triple_stage0:
                    s += f' + class{i}_stage0_adder{j*3+1}'
                if j*3+2 < num_triple_stage0:
                    s += f' + class{i}_stage0_adder{j*3+2}'
                print (f'{s};', file=fp) 
            print ('end', file=fp)
            print (file=fp)
            
            # stage 2 ... X
            curr_stage = 1
            curr_output_max = 9
            num_terms = num_triple_stage1
            while num_terms > 1:
                curr_output_max = curr_output_max * 2
                curr_output_num_bits = min(math.ceil(math.log2(curr_output_max)), target_output_bits)

                num_pair = math.ceil(num_terms / 2)
                for j in range(num_pair):
                    print (f'reg [{curr_output_num_bits-1}:0] class{i}_stage{curr_stage+1}_adder{j};', file=fp)
                print(file=fp)
                op = create_always_block(curr_stage+1)
                for j in range(num_pair):
                    if j*2+1 < num_terms:
                        print (f'\tclass{i}_stage{curr_stage+1}_adder{j} {op} class{i}_stage{curr_stage}_adder{j*2} + class{i}_stage{curr_stage}_adder{j*2+1};', file=fp)
                    else:
                        print (f'\tclass{i}_stage{curr_stage+1}_adder{j} {op} class{i}_stage{curr_stage}_adder{j*2};', file=fp)
                print ('end', file=fp)
                print (file=fp)
                curr_stage = curr_stage + 1
                num_terms = num_pair

            print (f'assign p[{i}] = class{i}_stage{curr_stage}_adder0;', file=fp)
            print (file=fp)

        print ('endmodule', file=fp)

    with open(f'{out_dir_path}/argmin.sv', 'w') as fp:
        print ('module vsalut_argmin (p, pred);', file=fp)
        print (file=fp)
        print (f'input [{math.ceil(math.log2(fhv_dimension*class_prune_level))-1}:0] p [0:{num_class-1}];', file=fp)
        print (f'output [0:{num_class-1}] pred;', file=fp)
        print (file=fp)

        for i in range(num_class):
            for j in range(num_class):
                if j > i:
                    print (f'wire cmp_{i}_lte_{j}; assign cmp_{i}_lte_{j} = p[{i}] <= p[{j}];', file=fp)
        print (file=fp)

        for i in range(num_class):
            term = []
            for j in range(num_class):
                if i == j:
                    pass
                elif i > j:
                    term.append(f'~cmp_{j}_lte_{i}')
                else:
                    term.append(f'cmp_{i}_lte_{j}')
            print (f"assign pred[{i}] = {' & '.join(term)};", file=fp)
        print (file=fp)

        print ('endmodule', file=fp)

    with open(f'{out_dir_path}/inference.sv', 'w') as fp:
        print ('module vsalut_inference (', file=fp)
        print ('\tinput clk,', file=fp) 
        print (f'\tinput [0:{vhv_dimension-1}] V [0:{num_input-1}],', file=fp) 
        print (f'\toutput reg [0:{num_class-1}] prediction', file=fp) 
        print (');', file=fp) 
        print (file=fp) 

        print (f'logic [0:{vhv_dimension-1}] V_q [0:{num_input-1}];', file=fp) 
        print (f'logic [0:{fhv_dimension-1}] s_d;', file=fp)
        print (f'logic [0:{fhv_dimension-1}] s_q;', file=fp)
        print (f'logic [{math.ceil(math.log2(fhv_dimension*class_prune_level))-1}:0] p_d [0:{num_class-1}];', file=fp)
        print (f'logic [{math.ceil(math.log2(fhv_dimension*class_prune_level))-1}:0] p_q [0:{num_class-1}];', file=fp)
        print (f'logic [0:{num_class-1}] pred;', file=fp)
        print (file=fp) 

        print ('vsalut_encode encode (V_q, s_d);', file=fp) 
        print (f"vsalut_similarity similarity ({'clk, ' if enable_add_register else ''}s_q, p_d);", file=fp)
        print ('vsalut_argmin argmin (p_q, pred);', file=fp)  
        print (file=fp)

        if not enable_input_register:
            print ('assign V_q = V;', file=fp)
        if not enable_s_register:
            print ('assign s_q = s_d;', file=fp)  
        if not enable_p_register:
            print ('assign p_q = p_d;', file=fp)  
        print (file=fp)

        print ('always @ (posedge clk) begin', file=fp)
        if enable_input_register:
            print (f'\tV_q <= V;', file=fp)
        if enable_s_register:
            print (f'\ts_q <= s_d;', file=fp)
        if enable_p_register:
            print (f'\tp_q <= p_d;', file=fp)
        print (f'\tprediction <= pred;', file=fp)
        print ('end', file=fp)
        print (file=fp)

        print ('endmodule', file=fp) 

    with open(f'{out_dir_path}/test_encode.sv', 'w') as fp:
        print ('module test_encode (', file=fp)
        print ('\tinput clk,', file=fp) 
        print (f'\tinput [0:{vhv_dimension-1}] V [0:{num_input-1}],', file=fp) 
        print (f'\toutput reg [0:{fhv_dimension-1}] s_q', file=fp) 
        print (');', file=fp) 
        print (file=fp) 

        print (f'logic [0:{vhv_dimension-1}] V_q [0:{num_input-1}];', file=fp) 
        print (f'logic [0:{fhv_dimension-1}] s_d;', file=fp)
        print (file=fp) 

        print ('vsalut_encode encode (V_q, s_d);', file=fp)
        print (file=fp)

        print ('always @ (posedge clk) begin', file=fp)
        print (f'\tV_q <= V;', file=fp)
        print (f'\ts_q <= s_d;', file=fp)
        print ('end', file=fp)
        print (file=fp)

        print ('endmodule', file=fp) 

    with open(f'{out_dir_path}/test_similarity.sv', 'w') as fp:
        print ('module test_similarity (', file=fp)
        print ('\tinput clk,', file=fp) 
        print (f'\tlogic [0:{fhv_dimension-1}] s_d,', file=fp) 
        print (f'\toutput reg [{math.ceil(math.log2(fhv_dimension*class_prune_level))-1}:0] p_q [0:{num_class-1}]', file=fp) 
        print (');', file=fp) 
        print (file=fp) 

        print (f'logic [0:{fhv_dimension-1}] s_q;', file=fp)
        print (f'logic [{math.ceil(math.log2(fhv_dimension*class_prune_level))-1}:0] p_d [0:{num_class-1}];', file=fp)
        print (file=fp) 

        print (f"vsalut_similarity similarity ({'clk, ' if enable_add_register else ''}s_q, p_d);", file=fp)
        print (file=fp)

        print ('always @ (posedge clk) begin', file=fp)
        print (f'\ts_q <= s_d;', file=fp)
        print (f'\tp_q <= p_d;', file=fp)
        print ('end', file=fp)
        print (file=fp)

        print ('endmodule', file=fp) 

    with open(f'{out_dir_path}/test_argmin.sv', 'w') as fp:
        print ('module test_argmin (', file=fp)
        print ('\tinput clk,', file=fp) 
        print (f'\tinput [{math.ceil(math.log2(fhv_dimension*class_prune_level))-1}:0] p_d [0:{num_class-1}],', file=fp) 
        print (f'\toutput reg [0:{num_class-1}] prediction', file=fp) 
        print (');', file=fp) 
        print (file=fp) 

        print (f'logic [{math.ceil(math.log2(fhv_dimension*class_prune_level))-1}:0] p_q [0:{num_class-1}];', file=fp)
        print (f'logic [0:{num_class-1}] pred;', file=fp)
        print (file=fp) 

        print ('vsalut_argmin argmin (p_q, pred);', file=fp)  
        print (file=fp)

        print ('always @ (posedge clk) begin', file=fp)
        print (f'\tp_q <= p_d;', file=fp)
        print (f'\tprediction <= pred;', file=fp)
        print ('end', file=fp)
        print (file=fp)

        print ('endmodule', file=fp) 

    print (f'Successfully generate output source at {out_dir_path}')
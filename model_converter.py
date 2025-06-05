import argparse
import os
import numpy as np

def numpy_to_hex_array(fp, arr, word_size, transpose=False, pack=True):
    # convert from -1/1 to 1/0
    arr = arr.astype(int)
    arr[arr == 1] = 0
    arr[arr == -1] = 1

    if pack:
        if word_size == 8:
            arr = np.packbits(arr, axis=1)
        elif word_size == 16:
            arr = np.packbits(arr, axis=1)
            arr = arr.view(np.uint16).byteswap()    # TODO: some systems might not need byteswap !!!
        elif word_size == 32:
            arr = np.packbits(arr, axis=1)
            arr = arr.view(np.uint32).byteswap()
        else:
            print('Error: word size should be 8, 16 or 32')
            exit(1)

    np.savetxt(fp, arr.T if transpose else arr, fmt='%#x', delimiter=',', newline=',\n')

def get_ctype(wordsize):
    if wordsize == 8 or wordsize == 16 or wordsize == 32:
        return f'uint{word_size}_t'
    else:
        print('Error: word size should be 8, 16 or 32')
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate model.h and model.c')
    parser.add_argument('-i', '--input-dir', help='path to the folder containing the F, V, and C vector')
    parser.add_argument('-o', '--output-dir', help='path to the dir to store the output model file')
    parser.add_argument('--output-filename', default='model', help='name of the output model source/header file')
    parser.add_argument('-n', '--model-name', default='', help='model name')
    parser.add_argument('-dv', '--v-dimension', type=int, default=4, help='dimension of the V vector (Dv in the paper)')
    parser.add_argument('--no-pack', action='store_true', help='')
    args = parser.parse_args()

    if ' ' in args.model_name:
        print ('Error: model name can\'t contain space')
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    F = np.load(os.path.join(args.input_dir, 'F.npy'))
    V = np.load(os.path.join(args.input_dir, 'V.npy'))
    C = np.load(os.path.join(args.input_dir, 'C.npy'))

    num_class = C.shape[0]
    fhv_dimension = F.shape[1]
    num_feature = F.shape[0]

    with open(os.path.join(args.output_dir, f'{args.output_filename}.h'), 'w') as outfile:
        outfile.write(f'#ifndef MODEL_{args.model_name.upper()}_H_\n')
        outfile.write(f'#define MODEL_{args.model_name.upper()}_H_\n\n')

        outfile.write(f'#include <stdint.h>\n')
        outfile.write(f'#include "microvsa_config.h"\n\n')

        outfile.write(f'#define MICROVSA_MODEL_FHV_DIMENSION_BIT {fhv_dimension}\n')
        outfile.write('#if MICROVSA_IMPL_WORDSIZE == 8\n')
        outfile.write(f'#define MICROVSA_MODEL_FHV_DIMENSION_WORD {fhv_dimension // 8}\n')
        outfile.write('#elif MICROVSA_IMPL_WORDSIZE == 16\n')
        outfile.write(f'#define MICROVSA_MODEL_FHV_DIMENSION_WORD {fhv_dimension // 16}\n')
        outfile.write('#elif MICROVSA_IMPL_WORDSIZE == 32\n')
        outfile.write(f'#define MICROVSA_MODEL_FHV_DIMENSION_WORD {fhv_dimension // 32}\n')
        outfile.write('#else\n')
        outfile.write('# error Unsupport look up configuration\n')
        outfile.write('#endif\n')
        outfile.write(f'#define MICROVSA_MODEL_NUM_CLASS {num_class}\n')
        outfile.write(f'#define MICROVSA_MODEL_NUM_FEATURE {num_feature}\n\n')

        outfile.write('#ifdef MODEL_F_IN_RAM\n')
        outfile.write('#define MODEL_F_QUALIFIER\n')
        outfile.write('#else\n')
        outfile.write('#define MODEL_F_QUALIFIER const\n')
        outfile.write('#endif\n')
        outfile.write('#ifdef MODEL_V_IN_RAM\n')
        outfile.write('#define MODEL_V_QUALIFIER\n')
        outfile.write('#else\n')
        outfile.write('#define MODEL_V_QUALIFIER const\n')
        outfile.write('#endif\n')
        outfile.write('#ifdef MODEL_C_IN_RAM\n')
        outfile.write('#define MODEL_C_QUALIFIER\n')
        outfile.write('#else\n')
        outfile.write('#define MODEL_C_QUALIFIER const\n')
        outfile.write('#endif\n\n')

        outfile.write('#if MICROVSA_IMPL_WORDSIZE == 8\n')
        outfile.write(f'extern MODEL_F_QUALIFIER uint8_t MICROVSA_MODEL_F[];\n')
        outfile.write(f'extern MODEL_V_QUALIFIER uint8_t MICROVSA_MODEL_V[];\n')
        outfile.write(f'extern MODEL_C_QUALIFIER uint8_t MICROVSA_MODEL_C[];\n')
        outfile.write('#elif MICROVSA_IMPL_WORDSIZE == 16\n')
        outfile.write(f'extern MODEL_F_QUALIFIER uint16_t MICROVSA_MODEL_F[];\n')
        outfile.write(f'extern MODEL_V_QUALIFIER uint16_t MICROVSA_MODEL_V[];\n')
        outfile.write(f'extern MODEL_C_QUALIFIER uint16_t MICROVSA_MODEL_C[];\n')
        outfile.write('#elif MICROVSA_IMPL_WORDSIZE == 32\n')
        outfile.write(f'extern MODEL_F_QUALIFIER uint32_t MICROVSA_MODEL_F[];\n')
        outfile.write(f'extern MODEL_V_QUALIFIER uint32_t MICROVSA_MODEL_V[];\n')
        outfile.write(f'extern MODEL_C_QUALIFIER uint32_t MICROVSA_MODEL_C[];\n')
        outfile.write('#else\n')
        outfile.write('# error Unsupport look up configuration\n')
        outfile.write('#endif\n\n')

        outfile.write('#endif')

    with open(os.path.join(args.output_dir, f'{args.output_filename}.c'), 'w') as outfile:
        outfile.write(f'#include "model.h"\n\n')

        outfile.write(f'#')
        for word_size in [8, 16, 32]:
            outfile.write(f'if MICROVSA_IMPL_WORDSIZE == {word_size}\n')

            outfile.write(f'MODEL_F_QUALIFIER {get_ctype(word_size)} MICROVSA_MODEL_F[] = {{\n')
            outfile.write('#ifndef MODEL_TRANSPOSE_F\n')
            numpy_to_hex_array(outfile, F, word_size, pack=not args.no_pack)
            outfile.write('#else\n')
            numpy_to_hex_array(outfile, F, word_size, transpose=True, pack=not args.no_pack)
            outfile.write('#endif\n')
            outfile.write('};\n')

            outfile.write(f'MODEL_V_QUALIFIER {get_ctype(word_size)} MICROVSA_MODEL_V[] = {{\n')
            numpy_to_hex_array(outfile, V[:, :word_size], word_size, pack=not args.no_pack)
            outfile.write('};\n')

            outfile.write(f'MODEL_C_QUALIFIER {get_ctype(word_size)} MICROVSA_MODEL_C[] = {{\n')
            outfile.write('#ifndef MODEL_TRANSPOSE_C\n')
            numpy_to_hex_array(outfile, C, word_size, pack=not args.no_pack)
            outfile.write('#else\n')
            numpy_to_hex_array(outfile, C, word_size, transpose=True, pack=not args.no_pack)
            outfile.write('#endif\n')
            outfile.write('};\n')

            outfile.write(f'#el')
        outfile.write('se\n')
        outfile.write('# error Unsupport look up configuration\n')
        outfile.write('#endif\n')

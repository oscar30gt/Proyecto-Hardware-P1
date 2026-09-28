            AREA code, CODE, READONLY
            PRESERVE8
                
            EXPORT dense_layer_q12_ARM_C
            IMPORT neuron_q12_C

; r0 = *input
; r1 = *weights
; r2 = *bias
; r3 = *output
; fp+8 = input_size
; fp+12 = output_size
; fp+16 = clamp_min
; fp+20 = clamp_max
dense_layer_q12_ARM_C
            PUSH {fp, lr}
            MOV fp, sp
            PUSH {v1-v7}
            
            MOV v1, #0              ; checksum = 0;
            
            LDR v2, [fp, #16]       ; clamp_min
            LDR v3, [fp, #20]       ; clamp_max
            PUSH {v2, v3}           ; push clamp_min and clamp_max for neuron_q12_C call

            MOV v2, r0              ; input pointer
            MOV v3, r1              ; weights pointer
            MOV v4, r2              ; bias pointer
            MOV v5, r3              ; output pointer
            LDRH v6, [fp, #8]       ; input_size

            LDR v7, [fp, #12]       ; output_size
            CMP v7, #0              ; if (output_size == 0) skip loop
            BEQ end_loop

loop        MOV r0, v2              ; r0 = input pointer
            MOV r1, v3              ; r1 = weights pointer
            MOV r2, v6              ; r2 = input_size
            LDRSH r3, [v4]          ; r3 = bias[o]
            BL neuron_q12_C         ; r0 = neuron_q12_C(...)
            STRH r0, [v5]           ; output[o] = y

            ADD v1, v1, v1, LSL #5  ; checksum = checksum * 33 (written as checksum * 32 + checksum)
            ADD v1, v1, r0          ; checksum = checksum + y

            ADD v3, v3, v6, LSL #1  ; iterate weights_o
            ADD v4, v4, #2          ; iterate bias
            ADD v5, v5, #2          ; iterate output pointer

            SUBS v7, v7, #1         ; for (uint16_t o = 0; o < output_size; ++o)
            BNE loop                ; branch to loop
            
end_loop    ADD sp, sp, #8          ; pop clamp_min and clamp_max         
            MOV r0, v1              ; return checksum
            POP {v1-v7, fp, pc}

            END
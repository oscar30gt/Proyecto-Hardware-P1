            AREA code, CODE, READONLY
            PRESERVE8
                
            EXPORT dense_layer_q12_ARM

; r0 = *input
; r1 = *weights
; r2 = *bias
; r3 = *output
; fp+8 = input_size
; fp+12 = output_size
; fp+16 = clamp_min
; fp+20 = clamp_max
dense_layer_q12_ARM
            PUSH {fp, lr}
            MOV fp, sp
            PUSH {v1-v7}
            
            MOV v1, #0              ; checksum = 0;
            
            ldr v2, [fp, #16]       ; clamp_min
            ldr v3, [fp, #20]       ; clamp_max
            PUSH {v2, v3}           ; push clamp_min and clamp_max for neuron_q12_C call

            mov v2, r0              ; input pointer
            mov v3, r1              ; weights pointer
            mov v4, r2              ; bias pointer
            mov v5, r3              ; output pointer
            ldrh v6, [fp, #8]       ; input_size

            ldr v7, [fp, #12]       ; output_size
            CMP v7, #0              ; if (output_size == 0) skip loop
            BEQ end_loop_l

loop_l    mov r0, v2                ; r0 = input pointer
            mov r1, v3              ; r1 = weights pointer
            mov r2, v6              ; r2 = input_size
            ldrsh r3, [v4]          ; r3 = bias[o]
            BL neuron_q12_ARM       ; r0 = neuron_q12_ARM(...)
            strh r0, [v5]           ; output[o] = y

            ADD v1, v1, v1, LSL #5  ; checksum = checksum * 33 (written as checksum * 32 + checksum)
            ADD v1, v1, r0          ; checksum = checksum + y

            ADD v3, v3, v6, LSL #1  ; iterate weights_o
            ADD v4, v4, #2          ; iterate bias
            ADD v5, v5, #2          ; iterate output pointer

            SUBS v7, v7, #1         ; for (uint16_t o = 0; o < output_size; ++o)
            BNE loop_l              ; branch to loop
            
end_loop_l  ADD sp, sp, #8          ; pop clamp_min and clamp_max         
            mov r0, v1              ; return checksum
            POP {v1-v7, fp, pc}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; r0 = *input
; r1 = *weights
; r2 = input_size
; r3 = bias
; fp+8 = clamp_min
; fp+12 = clamp_max
neuron_q12_ARM
            PUSH {fp, lr}
            MOV fp, sp
            PUSH {v1-v3}

            MOV v1, r3, LSL #12         ; v1 = bias << Q_SHIFT

            cmp r2, #0                  ; if (input_size == 0) skip loop
            BEQ end_loop_n

loop_n      ldrsh v2, [r0], #2          ; v2 = input[i]; input++
            ldrsh v3, [r1], #2          ; v3 = weights[i]; weights++
            MLA v1, v3, v2, v1          ; acc += weights[i] * input[i]

            SUBS r2, r2, #1             ; for (uint16_t i = 0; i < input_size; ++i)
            BNE loop_n                  ; branch to loop

end_loop_n  MOV r0, v1, ASR #12         ; r0 = acc >> Q_SHIFT
            ldrsh r1, [fp, #8]          ; r1 = clamp_min
            ldrsh r2, [fp, #12]         ; r2 = clamp_max

            CMP r0, r1                  ; if (y < clamp_min) return clamp_min
            MOVLT r0, r1
            CMP r0, r2                  ; if (y > clamp_max) return clamp_max
            MOVGT r0, r2
            
            POP {v1-v3, fp, pc}

            END
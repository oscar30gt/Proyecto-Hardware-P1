            AREA code, CODE, READONLY
            PRESERVE8

            EXPORT dense_layer_q12_ARM
            EXPORT neuron_q12_ARM

; r0 = *input
; r1 = *weights
; r2 = *bias
; r3 = *output
; sp+36 = input_size        (offsets after PUSH {v1-v8, lr})
; sp+40 = output_size
; sp+44 = clamp_min
; sp+48 = clamp_max
dense_layer_q12_ARM
            PUSH {v1-v8, lr}        ; neuron inlined -> leaf function, no call, SP does not need 8-byte alignment

            MOV v1, #0              ; checksum = 0;

            ADD fp, sp, #36         ; fp = address of the stacked arguments
            LDMIA fp, {v6-v8, lr}   ; v6 = input_size, v7 = output_size, v8 = clamp_min, lr = clamp_max

            mov v2, r0              ; input pointer
            mov v3, r1              ; weights pointer
            mov v4, r2              ; bias pointer
            mov v5, r3              ; output pointer

            CMP v7, #0              ; if (output_size == 0) skip loop
            BEQ end_loop_l

loop_l      ldrsh r3, [v4], #2      ; r3 = bias[o]; bias++
            MOV r3, r3, LSL #12     ; r3 = acc = bias << Q_SHIFT
            mov r0, v2              ; r0 = input pointer (restart for every neuron)

            MOVS r2, v6, LSR #1     ; r2 = input_size / 2 (pairs); C = input_size odd
            BCC pairs_i             ; if (input_size even) skip single element
            ldrsh fp, [r0], #2      ; fp = input[i]; input++
            ldrsh r1, [v3], #2      ; r1 = weights[i]; weights++
            MLA r3, r1, fp, r3      ; acc += weights[i] * input[i]

pairs_i     BEQ end_loop_i          ; if (no pairs) skip loop (flags still from MOVS)

loop_i      ldrsh fp, [r0], #2      ; fp = input[i]; input++
            ldrsh r1, [v3], #2      ; r1 = weights[i]; weights++
            MLA r3, r1, fp, r3      ; acc += weights[i] * input[i]
            ldrsh fp, [r0], #2      ; fp = input[i+1]; input++
            ldrsh r1, [v3], #2      ; r1 = weights[i+1]; weights++
            MLA r3, r1, fp, r3      ; acc += weights[i+1] * input[i+1]

            SUBS r2, r2, #1         ; for (uint16_t i = 0; i < input_size; i += 2)
            BNE loop_i              ; branch to loop

end_loop_i  MOV r3, r3, ASR #12     ; y = acc >> Q_SHIFT

            CMP r3, v8              ; if (y < clamp_min) y = clamp_min
            MOVLT r3, v8
            CMP r3, lr              ; if (y > clamp_max) y = clamp_max
            MOVGT r3, lr
            strh r3, [v5], #2       ; output[o] = y; output++

            LSL r3, r3, #16         ; (uint16_t)y: shift for checksum calculation
            ADD v1, v1, v1, LSL #5  ; checksum = checksum * 33 (written as checksum * 32 + checksum)
            ADD v1, v1, r3, LSR #16 ; checksum = checksum + (uint16_t)y

            SUBS v7, v7, #1         ; for (uint16_t o = 0; o < output_size; ++o)
            BNE loop_l              ; branch to loop

end_loop_l  mov r0, v1              ; return checksum
            POP {v1-v8, lr}
            BX lr


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; PARA SER LLAMADA POR C. EN LA VERSION OPTIMIZADA, ARM NO LLAMA A ESTA SUBRUTINA

; r0 = *input
; r1 = *weights
; r2 = input_size
; r3 = bias
; sp+4 = clamp_min          (offsets after PUSH {v1})
; sp+8 = clamp_max
neuron_q12_ARM
            PUSH {v1}                   ; leaf function: only one extra register needed

            MOV r3, r3, LSL #12         ; r3 = acc = bias << Q_SHIFT

            MOVS r2, r2, LSR #1         ; r2 = input_size / 2 (pairs); C = input_size odd
            BCC pairs_n                 ; if (input_size even) skip single element
            ldrsh fp, [r0], #2          ; fp = input[i]; input++
            ldrsh v1, [r1], #2          ; v1 = weights[i]; weights++
            MLA r3, v1, fp, r3          ; acc += weights[i] * input[i]

pairs_n     BEQ end_loop_n              ; if (no pairs) skip loop (flags still from MOVS)

loop_n      ldrsh fp, [r0], #2          ; fp = input[i]; input++
            ldrsh v1, [r1], #2          ; v1 = weights[i]; weights++
            MLA r3, v1, fp, r3          ; acc += weights[i] * input[i]
            ldrsh fp, [r0], #2          ; fp = input[i+1]; input++
            ldrsh v1, [r1], #2          ; v1 = weights[i+1]; weights++
            MLA r3, v1, fp, r3          ; acc += weights[i+1] * input[i+1]

            SUBS r2, r2, #1             ; for (uint16_t i = 0; i < input_size; i += 2)
            BNE loop_n                  ; branch to loop

end_loop_n  MOV r0, r3, ASR #12         ; r0 = acc >> Q_SHIFT
            LDMIB sp, {r1, r2}          ; r1 = clamp_min, r2 = clamp_max

            CMP r0, r1                  ; if (y < clamp_min) return clamp_min
            MOVLT r0, r1
            CMP r0, r2                  ; if (y > clamp_max) return clamp_max
            MOVGT r0, r2

            POP {v1}
            BX lr

            END
				
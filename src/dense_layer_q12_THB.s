            AREA code, CODE, READONLY
            PRESERVE8

            EXPORT dense_layer_q12_THB
            IMPORT neuron_q12_C

; r0 = *input       ; saved at r7+0
; r1 = *weights     ; saved at r7+4
; r2 = *bias        ; r6
; r3 = *output      ; saved at r7+8
; r7+32 = input_size
; r7+36 = output_size
; r7+40 = clamp_min
; r7+44 = clamp_max
; (r7 = sp after PUSH {r0, r1, r3}: 12 bytes + 20 bytes of PUSH {r4-r7, lr} = 32)
dense_layer_q12_THB
            PUSH {r4-r7, lr}

            LDR r4, =thb            ; thumb part @
            ORR r4, r4, #1          ; set thumb bit
            BX r4                   ; THUMB mode

            THUMB
thb         PUSH {r0, r1, r3}
            MOV r7, sp

            MOVS r6, r2             ; r6 = bias pointer

            LDR r0, [r7, #40]       ; clamp_min
            LDR r1, [r7, #44]       ; clamp_max
            PUSH {r0, r1}           ; push clamp_min and clamp_max for neuron_q12_C call

            MOVS r4, #0             ; checksum = 0;

            LDR r5, [r7, #36]       ; output_size
            CMP r5, #0              ; if (output_size == 0) skip loop
            BEQ end_loop_l

loop_l      LDR r0, [r7, #0]        ; r0 = input pointer
            LDR r1, [r7, #4]        ; r1 = weights pointer
            LDR r2, [r7, #32]       ; r2 = input_size

            LSLS r3, r2, #1         ; Iterate weights
            ADDS r3, r1, r3
            STR r3, [r7, #4]

            MOVS r3, #0
            LDRSH r3, [r6, r3]      ; r3 = bias[o]
            BL neuron_q12_C         ; r0 = neuron_q12_C(...)
            LDR r1, [r7, #8]        ; r1 = output pointer
            STRH r0, [r1, #0]       ; output[o] = y
            ADDS r1, r1, #2         ; output pointer += 2
            STR r1, [r7, #8]        ; store updated output pointer

            LSLS r0, r0, #16
            LSRS r0, r0, #16        ; (uint16_t)y
            MOVS r1, r4
            LSLS r1, r1, #5         ; checksum * 32
            ADDS r4, r4, r1         ; checksum = checksum * 33 (written as checksum * 32 + checksum)
            ADDS r4, r4, r0         ; checksum = checksum + y

            ADDS r6, r6, #2         ; iterate bias

            SUBS r5, r5, #1         ; for (uint16_t o = 0; o < output_size; ++o)
            BNE loop_l              ; branch to loop

end_loop_l  ADD sp, sp, #20         ; pop clamp_min, clamp_max and the saved r0, r1, r3
            MOVS r0, r4             ; return checksum

            POP {r4-r7}
            POP {r3}                ; saved lr (POP {pc} would not switch state on ARMv4T)
            BX r3

            END

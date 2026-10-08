            AREA code, CODE, READONLY
            PRESERVE8

            EXPORT dense_layer_q12_THB
            EXPORT neuron_q12_THB

; r0 = *input       ; saved at r7+0
; r1 = *weights     ; saved at r7+4
; r2 = *bias        ; r6
; r3 = *output      ; saved at r7+8
; r7+32 = input_size
; r7+36 = output_size
; r7+40 = clamp_min
; r7+44 = clamp_max
dense_layer_q12_THB
            PUSH {r4-r7, lr}

            LDR r4, =thb            ; thumb part @
            ORR r4, r4, #1          ; set thumb bit
            BX r4                   ; THUMB mode
			LTORG

            THUMB
thb         PUSH {r0, r1, r3}
            MOV r7, sp

            MOVS r6, r2             ; r6 = bias pointer

            LDR r0, [r7, #40]       ; clamp_min
            LDR r1, [r7, #44]       ; clamp_max
            PUSH {r0, r1}           ; push clamp_min and clamp_max for neuron_q12_THB call

            MOVS r4, #0             ; checksum = 0;

            LDRH r5, [r7, #36]      ; output_size
            CMP r5, #0              ; if (output_size == 0) skip loop
            BEQ end_loop_l

loop_l      LDR r0, [r7, #0]        ; r0 = input pointer
            LDR r1, [r7, #4]        ; r1 = weights pointer
            LDRH r2, [r7, #32]      ; r2 = input_size

            LSLS r3, r2, #1         ; Iterate weights
            ADDS r3, r1, r3
            STR r3, [r7, #4]

            MOVS r3, #0
            LDRSH r3, [r6, r3]      ; r3 = bias[o]
            BL neuron_q12_THB       ; r0 = neuron_q12_THB(...)
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
            POP {r3}                ; saved lr (POP {pc} would not switch state on ARM)
            BX r3
			
; r0 = *input
; r1 = *weights
; r2 = n (input_size)
; r3 = bias           -> se reutiliza como acumulador acc
; [sp+0]  = clamp_min (a la entrada)  -> sp+16 tras el PUSH
; [sp+4]  = clamp_max (a la entrada)  -> sp+20 tras el PUSH
; r4, r5 = temporales (input[i], weights[i])
; r6 = offset en bytes, recorre los vectores hacia atras (2*(n-1) ... 0)
neuron_q12_THB
            PUSH {r4-r6, lr}

            LSLS r3, r3, #12        ; acc = bias << Q_SHIFT

            LSLS r6, r2, #1         ; r6 = 2*n
            SUBS r6, r6, #2         ; r6 = 2*(n-1); si n == 0 queda negativo
            BMI end_loop_n          ; if (n == 0) skip loop

loop_n      LDRSH r4, [r0, r6]      ; r4 = input[i]
            LDRSH r5, [r1, r6]      ; r5 = weights[i]
            MULS r4, r5, r4         ; r4 = weights[i] * input[i]
            ADDS r3, r3, r4         ; acc += weights[i] * input[i]
            SUBS r6, r6, #2         ; i-- (recorrido inverso)
            BPL loop_n              ; while (i >= 0)

end_loop_n  ASRS r0, r3, #12        ; r0 = acc >> Q_SHIFT

            LDR r1, [sp, #16]       ; r1 = clamp_min
            CMP r0, r1              ; if (y < clamp_min) y = clamp_min
            BGE chk_max
            MOVS r0, r1
chk_max     LDR r1, [sp, #20]       ; r1 = clamp_max
            CMP r0, r1              ; if (y > clamp_max) y = clamp_max
            BLE end_neuron
            MOVS r0, r1

end_neuron  POP {r4-r6}
            POP {r1}                ; lr guardado (POP {pc} no cambia de estado en ARM)
            BX r1                   ; vuelve a ARM o Thumb segun el bit 0 (llamador C en ARM)

            END
				
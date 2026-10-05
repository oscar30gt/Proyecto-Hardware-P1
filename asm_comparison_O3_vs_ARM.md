# Compiler (`-O3 -Otime`) vs hand-written ARM: `neuron_q12` and `dense_layer_q12`

Target: **LPC2105 / ARM7TDMI** (ARMv4T, ARM state, `--apcs=interwork`).
Source: `disassemby.txt`. The listings below have been cleaned up: interleaved C source removed, immediates in decimal, `STMDB R13!` / `LDMIA R13!` written as `PUSH` / `POP`, and comments added.

### Cycle model used for the estimates (ARM7TDMI TRM, zero wait states)

| Instruction | Cycles |
|---|---|
| ALU op (`MOV`, `ADD`, `CMP`, …), branch not taken, failed condition | 1 |
| `LDR` / `LDRH` / `LDRSH` | 3 |
| `STR` / `STRH` | 2 |
| `B` / `BL` taken, `BX` | 3 |
| `MUL` / `MLA` | 1 + m (+1 for MLA). m = 1 if \|Rs\| < 2^8, m = 2 if \|Rs\| < 2^16. For Q12 data usually m = 2, so **MLA ≈ 4** |
| `LDM` n regs / `LDM` n regs including PC | n + 2 / n + 4 |
| `STM` n regs | n + 1 |

These are estimates. Flash wait states and the MAM change the absolute numbers. To get real figures, use `medidas.c`: `ciclos[0]` (C), `ciclos[4]` (ARM), and `ciclos[3]` (ARM layer + C neuron). Comparing `ciclos[3]` with `ciclos[4]` isolates the neuron, because the outer loop is the same in both.

---

## 1. `neuron_q12`

### 1.1 C source

```c
int32_t acc = ((int32_t)bias_q12) << Q_SHIFT;
for (uint16_t i = 0; i < n; ++i)
    acc += (int32_t)input[i] * (int32_t)weights[i];
int32_t result_q12 = acc >> Q_SHIFT;
return clamp_i16_q12(result_q12, clamp_min, clamp_max);   // lo / hi / (int16_t)x
```

### 1.2 Compiler output: `neuron_q12_C` @ `0x1C44`

(This standalone copy is only used by `dense_layer_q12_ARM_C`; `dense_layer_q12_C` has its own inlined copy, see section 2.)

Registers: `R11` = input, `R1` = weights, `R2` = n, `R3` = acc, `R6` = second accumulator, `R10` = clamp_min, `R0` = clamp_max.

```asm
; ---------- prologue: saves 9 registers ----------
1C44  PUSH    {R4-R11,R14}
1C48  MOV     R11, R0               ; R11 = input
1C4C  LDR     R0,  [R13,#40]        ; R0  = clamp_max   (stack arg, loaded as a word)
1C50  LDR     R10, [R13,#36]        ; R10 = clamp_min
1C54  SUB     R5, R2, #1            ; R5  = n-1
1C58  CMP     R5, #0
1C5C  MOV     R3, R3, LSL #12       ; acc = bias << 12
1C60  BLE     1CB8                  ; n <= 1  -> go straight to the tail element

; ---------- loop set-up: unrolled x2, handles parity ----------
1C64  TST     R2, #1                ; n odd?
1C68  SUB     R4,  R11, #2          ; R4  = input   - 1 element  (for pre-indexed !)
1C6C  SUB     R12, R1,  #2          ; R12 = weights - 1 element
1C70  BNE     1C80                  ; odd -> skip peeled element
1C74  LDRSH   R6, [R4,#2]!          ;   even: peel element 0
1C78  LDRSH   R7, [R12,#2]!
1C7C  MLA     R3, R6, R7, R3
1C80  LDRSH   R7, [R4,#2]           ; software pipelining: pre-load next pair
1C84  LDRSH   R8, [R12,#2]
1C88  MOVS    R5, R5, ASR #1        ; iterations = (n-1)/2
1C8C  MOV     R6, #0                ; acc2 = 0
1C90  BEQ     1CB4

; ---------- inner loop: 2 MACs per iteration ----------
1C94  LDRSH   R14, [R4,#4]!         ; load element i+1
1C98  LDRSH   R9,  [R12,#4]!
1C9C  MLA     R3, R7, R8, R3        ; acc  += element i   (loaded in previous iteration)
1CA0  MLA     R6, R14, R9, R6       ; acc2 += element i+1
1CA4  LDRSH   R7, [R4,#2]           ; pre-load element i+2
1CA8  LDRSH   R8, [R12,#2]
1CAC  SUBS    R5, R5, #1
1CB0  BNE     1C94
1CB4  ADD     R3, R3, R6            ; acc += acc2

; ---------- tail: last element (n-1) ----------
1CB8  SUBS    R2, R2, #1            ; R2 = n-1
1CBC  BMI     1CD4                  ; n == 0 -> no tail
1CC0  ADD     R12, R11, R2, LSL #1  ; &input[n-1]
1CC4  ADD     R1,  R1,  R2, LSL #1  ; &weights[n-1]
1CC8  LDRSH   R12, [R12]            ; (already in R7/R8 on the main path, loaded again)
1CCC  LDRSH   R1,  [R1]
1CD0  MLA     R3, R12, R1, R3

; ---------- >>12 and clamp ----------
1CD4  MOV     R1, R3, ASR #12       ; y = acc >> 12
1CD8  CMP     R1, R10
1CDC  MOVLT   R0, R10               ; y < min -> return min
1CE0  POPLT   {R4-R11,R14}
1CE4  BXLT    R14
1CE8  POP     {R4-R11,R14}
1CEC  CMP     R0, R1                ; max >= y ?
1CF0  MOVGE   R0, R1, LSL #16       ;   yes: return (int16_t)y
1CF4  MOVGE   R0, R0, ASR #16
1CF8  BX      R14                   ;   no: R0 already holds max
```

**Total: 46 instructions.**

### 1.3 My implementation: `neuron_q12_ARM` @ `0x02DC`

```asm
; ---------- prologue: frame pointer + 3 registers ----------
02DC  PUSH    {R11,R14}             ; PUSH {fp, lr}
02E0  MOV     R11, R13              ; fp = sp
02E4  PUSH    {R4-R6}               ; v1-v3
02E8  MOV     R4, R3, LSL #12       ; acc = bias << 12
02EC  CMP     R2, #0
02F0  BEQ     0308                  ; n == 0 -> skip loop

; ---------- inner loop: 1 MAC per iteration ----------
02F4  LDRSH   R5, [R0], #2          ; input[i],   post-increment
02F8  LDRSH   R6, [R1], #2          ; weights[i], post-increment
02FC  MLA     R4, R6, R5, R4        ; acc += w * x
0300  SUBS    R2, R2, #1
0304  BNE     02F4

; ---------- >>12 and clamp ----------
0308  MOV     R0, R4, ASR #12       ; y = acc >> 12
030C  LDRSH   R1, [R11,#8]          ; clamp_min
0310  LDRSH   R2, [R11,#12]         ; clamp_max
0314  CMP     R0, R1
0318  MOVLT   R0, R1
031C  CMP     R0, R2
0320  MOVGT   R0, R2
0324  POP     {R4-R6,R11,PC}
```

**Total: 19 instructions.**

### 1.4 Comparison

| | `neuron_q12_C` (O3-Otime) | `neuron_q12_ARM` (mine) |
|---|---|---|
| Instructions (code size) | 46 (184 B) | **19 (76 B)** |
| Registers saved | 9 (R4-R11, LR) | 5 (R4-R6, FP, LR) |
| Unrolling | x2, peeled element when n is even, separate tail element | none |
| Accumulators | 2 (R3 + R6) | 1 |
| Inner loop cost | 24 cyc / 2 elements = **12 cyc/elem** | 14 cyc / 1 element = **14 cyc/elem** |
| Fixed overhead (outside the loop) | ~55 cyc | **~31 cyc** |
| **Estimated total, n = 8** | **~151 cyc** | **~141 cyc** |
| Return | `BX LR` (interworking-safe) | `POP {…,PC}` (ARM callers only on ARMv4T) |

**What the compiler does better**
1. **x2 unrolling.** The `SUBS` + `BNE` (4 cycles) is paid once per two elements, saving about 2 cycles per element. This is the only real gain, and it only pays off once n is large enough to cover the extra set-up.
2. **Clamp values are loaded as words (`LDR`)**, not `LDRSH`. AAPCS already makes the caller sign- or zero-extend stacked sub-word arguments, so this is safe.
3. **No frame pointer.** Stack arguments are read relative to SP.
4. **Returns with `BX LR`**, so a Thumb caller would work.

**What my version does better**
1. **Much smaller prologue and epilogue.** The compiler saves 9 registers (about 10 + 11 cycles) because of the second accumulator and the software pipelining. My version saves 5.
2. **Post-indexed loads (`[R0], #2`).** The compiler has to set pointers to `ptr - 2` and use pre-indexed `!` addressing.
3. **No redundant work.** On the main path the compiler loads `input[n-1]` and `weights[n-1]` twice: once at the end of the loop (`1CA4`/`1CA8`), then again in the tail (`1CC8`/`1CCC`). That costs about 8 wasted cycles.
4. **No `(int16_t)` sign extension** (`LSL #16` / `ASR #16`). The clamp already guarantees that y fits in int16, and the compiler does not see that.
5. **Software pipelining and the second accumulator buy nothing on ARM7TDMI.** The core has no load-use or multiply-result interlocks, so a load result or an MLA result can be used by the next instruction at no extra cost. Those transformations help cores such as ARM9E, but here they only cost registers, which means a bigger PUSH/POP.

For `INPUT_SIZE = 8` the two versions end up close, and the compiler's unrolling only wins for larger n. Unrolling while keeping my lean prologue gets the best of both.

### 1.5 Recommendations for `neuron_q12_ARM`

1. **Unroll x2 (or x4)** and handle an odd n with one peeled element. Use `MOVS rX, n, LSR #1`, which gives the pair count and also puts n's low bit in C and "no pairs" in Z with a single instruction.
2. **Drop the frame pointer.** Stack arguments can be addressed from SP.
3. **Use the scratch registers (`R3`, `R12`)** instead of callee-saved ones. Once `bias << 12` is computed, `R3` is free to hold the accumulator. That leaves one callee-saved register to push.
4. **Load both clamp values with one `LDMIB sp, {r1, r2}`** (4 cycles instead of 2 × 3). This relies on the caller extending the values to words, which AAPCS requires and the C compiler itself relies on (`1C4C`/`1C50`).
5. **Return with `BX LR`** so the function stays interworking-safe.
6. *(Minor)* MLA timing depends on the **Rs** operand (the third one). If one operand is usually small (for example |weights| < 128), put it in Rs so that m = 1.

```asm
; r0=*input  r1=*weights  r2=n  r3=bias   [sp+4]=clamp_min  [sp+8]=clamp_max (after the PUSH)
neuron_q12_ARM
            PUSH    {r4}                    ; leaf function: one scratch register is enough
            MOV     r3, r3, LSL #12         ; acc = bias << 12      (reuse r3)
            MOVS    r2, r2, LSR #1          ; r2 = n/2 ; C = n odd ; Z = (n/2 == 0)
            BCC     n_pairs                 ; even n -> no peeled element
            LDRSH   r12, [r0], #2           ; odd n: peel one element
            LDRSH   r4,  [r1], #2
            MLA     r3, r12, r4, r3
n_pairs     BEQ     n_done                  ; flags are still those of MOVS
n_loop      LDRSH   r12, [r0], #2
            LDRSH   r4,  [r1], #2
            MLA     r3, r12, r4, r3
            LDRSH   r12, [r0], #2
            LDRSH   r4,  [r1], #2
            MLA     r3, r12, r4, r3
            SUBS    r2, r2, #1
            BNE     n_loop
n_done      MOV     r0, r3, ASR #12         ; y = acc >> 12
            LDMIB   sp, {r1, r2}            ; r1 = clamp_min, r2 = clamp_max
            CMP     r0, r1
            MOVLT   r0, r1
            CMP     r0, r2
            MOVGT   r0, r2
            POP     {r4}
            BX      lr
```

Estimate for n = 8: **~117 cycles**, against ~141 for the current version and ~151 for the compiler's. The code is 25 instructions.
With x4 unrolling the loop drops to about 11 cyc/elem, but the remainder (n mod 4) needs more handling.

---

## 2. `dense_layer_q12`

### 2.1 C source

```c
uint32_t checksum = 0;
for (uint16_t o = 0; o < output_size; ++o) {
    const int16_t *weights_o = &weights[o * input_size];
    int16_t y = neuron_q12_C(input, weights_o, input_size, bias[o], clamp_min, clamp_max);
    output[o] = y;
    checksum = checksum * 33u + (uint16_t)y;
}
return checksum;
```

### 2.2 Compiler output: `dense_layer_q12_C` @ `0x044C`

**The compiler inlined `neuron_q12_C`.** There is no `BL` in this function. That removes the call overhead, but with the neuron's registers added it runs out of registers and **spills to the stack**.

Stack frame after the prologue (64 bytes):

| SP offset | Content |
|---|---|
| `+0x00` | spill: `weights_o` |
| `+0x04` | spill: `input` |
| `+0x08` | spill: `input_size - 1` |
| `+0x0C … +0x18` | saved R0-R3 = `input`, `weights`, `bias`, `output` (used as memory variables) |
| `+0x1C … +0x3C` | saved R4-R11, LR |
| `+0x40 … +0x4C` | stack args: `input_size`, `output_size`, `clamp_min`, `clamp_max` |

Registers: `R0` = checksum, `R4` = o, `R11` = input_size, `R10` = clamp_min, `R14` = clamp_max, `R1` = acc.

```asm
; ---------- prologue: saves 13 registers (R0-R3 used as spill slots) ----------
044C  PUSH    {R0-R11,R14}
0450  SUB     R13, R13, #12
0454  ADD     R14, R13, #68
0458  LDMIA   R14, {R1,R10,R14}     ; R1 = output_size, R10 = clamp_min, R14 = clamp_max  (one LDM)
045C  LDR     R11, [R13,#64]        ; R11 = input_size
0460  CMP     R1, #0
0464  SUBHI   R1, R11, #1
0468  MOV     R0, #0                ; checksum = 0
046C  STRHI   R1, [R13,#8]          ; spill n-1
0470  MOV     R4, R0                ; o = 0
0474  ADDLS   R13, R13, #28         ; output_size == 0 -> early return
0478  POPLS   {R4-R11,R14}
047C  BXLS    R14

; ================= OUTER LOOP (per neuron) =================
0480  MUL     R2, R4, R11           ; o * input_size     (recomputed every iteration)
0484  LDR     R1, [R13,#16]         ; weights            (reload from stack)
0488  ADD     R1, R1, R2, LSL #1    ; weights_o
048C  LDR     R2, [R13,#12]         ; input              (reload from stack)
0490  STMIA   R13, {R1,R2}          ; spill weights_o, input
0494  LDR     R1, [R13,#20]         ; bias               (reload from stack)
0498  LDR     R12, [R13,#8]         ; n-1                (reload from stack)
049C  ADD     R1, R1, R4, LSL #1
04A0  LDRSH   R1, [R1]              ; bias[o]
04A4  CMP     R12, #0
04A8  MOV     R1, R1, LSL #12       ; acc = bias << 12
04AC  BLE     0508                  ; n <= 1 -> tail only

;   ---- inlined neuron: set-up, same scheme as 1.2 ----
04B0  SUB     R3, R2, #2            ; R3 = input - 1
04B4  LDR     R2, [R13]             ; weights_o          (reload of a value just stored)
04B8  TST     R11, #1               ; parity test, repeated for every neuron (loop-invariant)
04BC  SUB     R2, R2, #2
04C0  BNE     04D0
04C4  LDRSH   R5, [R3,#2]!          ;   even: peel element 0
04C8  LDRSH   R6, [R2,#2]!
04CC  MLA     R1, R5, R6, R1
04D0  LDRSH   R6, [R3,#2]           ; pre-load
04D4  LDRSH   R7, [R2,#2]
04D8  MOVS    R12, R12, ASR #1
04DC  MOV     R5, #0                ; acc2 = 0
04E0  BEQ     0504

;   ---- inner loop: 2 MACs per iteration ----
04E4  LDRSH   R8, [R3,#4]!
04E8  LDRSH   R9, [R2,#4]!
04EC  MLA     R1, R6, R7, R1
04F0  MLA     R5, R8, R9, R5
04F4  LDRSH   R6, [R3,#2]
04F8  LDRSH   R7, [R2,#2]
04FC  SUBS    R12, R12, #1
0500  BNE     04E4
0504  ADD     R1, R1, R5            ; acc += acc2

;   ---- tail element (n-1): reloads pointers from the stack ----
0508  SUBS    R2, R11, #1
050C  BMI     052C
0510  LDR     R3, [R13,#4]          ; input              (reload)
0514  ADD     R3, R3, R2, LSL #1
0518  LDRSH   R12, [R3]             ; (already in R6 on the main path, loaded again)
051C  LDR     R3, [R13]             ; weights_o          (reload)
0520  ADD     R2, R3, R2, LSL #1
0524  LDRSH   R2, [R2]
0528  MLA     R1, R12, R2, R1

;   ---- >>12 and clamp ----
052C  MOV     R1, R1, ASR #12       ; y
0530  CMP     R1, R10
0534  MOV     R3, R14               ; useless copy of clamp_max
0538  MOVLT   R1, R10               ; y < min -> min
053C  BLT     0550
0540  CMP     R3, R1
0544  MOVGE   R1, R1, LSL #16       ; (int16_t)y ...
0548  MOVLT   R1, R14               ; y > max -> max
054C  MOVGE   R1, R1, ASR #16       ; ... (int16_t)y

;   ---- store + checksum ----
0550  LDR     R2, [R13,#24]         ; output             (reload)
0554  ADD     R0, R0, R0, LSL #5    ; checksum *= 33     (same trick as mine)
0558  ADD     R2, R2, R4, LSL #1
055C  STRH    R1, [R2]              ; output[o] = y
0560  MOV     R1, R1, LSL #16
0564  MOV     R1, R1, LSR #16       ; (uint16_t)y
0568  ADD     R0, R0, R1            ; checksum += (uint16_t)y
056C  ADD     R1, R4, #1
0570  BIC     R4, R1, #0x10000      ; o = (uint16_t)(o+1)  <- cost of the uint16_t counter
0574  LDR     R1, [R13,#68]         ; output_size        (reload)
0578  CMP     R4, R1
057C  BCC     0480
; ===========================================================

0580  ADD     R13, R13, #28
0584  POP     {R4-R11,R14}
0588  BX      R14
```

**Total: 80 instructions.**

### 2.3 My implementation: `dense_layer_q12_ARM` @ `0x0254`

```asm
; ---------- prologue ----------
0254  PUSH    {R11,R14}             ; PUSH {fp, lr}
0258  MOV     R11, R13              ; fp = sp
025C  PUSH    {R4-R10}              ; v1-v7
0260  SUB     R13, R13, #4          ; keep SP 8-byte aligned
0264  MOV     R4, #0                ; checksum = 0
0268  LDR     R5, [R11,#16]         ; clamp_min
026C  LDR     R6, [R11,#20]         ; clamp_max
0270  PUSH    {R5,R6}               ; stack args for neuron_q12_ARM (pushed once, reused)
0274  MOV     R5, R0                ; input
0278  MOV     R6, R1                ; weights_o
027C  MOV     R7, R2                ; bias ptr
0280  MOV     R8, R3                ; output ptr
0284  LDRH    R9,  [R11,#8]         ; input_size
0288  LDRH    R10, [R11,#12]        ; output_size (down-counter)
028C  CMP     R10, #0
0290  BEQ     02D0

; ================= OUTER LOOP (per neuron) =================
0294  MOV     R0, R5                ; input
0298  MOV     R1, R6                ; weights_o
029C  MOV     R2, R9                ; input_size
02A0  LDRSH   R3, [R7]              ; bias[o]
02A4  BL      neuron_q12_ARM        ; -> 19-instruction neuron, see 1.3
02A8  STRH    R0, [R8]              ; output[o] = y
02AC  MOV     R0, R0, LSL #16
02B0  MOV     R0, R0, LSR #16       ; (uint16_t)y
02B4  ADD     R4, R4, R4, LSL #5    ; checksum *= 33
02B8  ADD     R4, R4, R0            ; checksum += (uint16_t)y
02BC  ADD     R6, R6, R9, LSL #1    ; weights_o += input_size   (strength-reduced, no MUL)
02C0  ADD     R7, R7, #2            ; bias++
02C4  ADD     R8, R8, #2            ; output++
02C8  SUBS    R10, R10, #1
02CC  BNE     0294
; ===========================================================

02D0  ADD     R13, R13, #12         ; drop clamp args + padding
02D4  MOV     R0, R4                ; return checksum
02D8  POP     {R4-R11,PC}
```

**Total: 34 instructions (+19 for the neuron = 53).**

### 2.4 Comparison

| | `dense_layer_q12_C` (O3-Otime) | `dense_layer_q12_ARM` (mine) |
|---|---|---|
| Instructions | 80 (neuron inlined) | **34 + 19 = 53** |
| Calls per neuron | **0 (inlined)** | 1 `BL` + full neuron prologue and epilogue |
| Stack traffic per neuron (excluding the call) | 9 `LDR` + 1 `STM` (spills) | **0** |
| `weights_o` computation | `MUL` + `ADD` every iteration | **`ADD …, LSL #1`** (strength-reduced) |
| Loop counter | `uint16_t` up-counter: `ADD`+`BIC`+`LDR`+`CMP` | **down-counter `SUBS`** |
| Loop-invariant work in the loop | parity test, reload of n-1 | none |
| Clamp values | in registers (R10, R14) | reloaded from the stack in every neuron call |
| Est. cycles per neuron (n = 8) | ~170 | **~163** (22 outer + ~141 neuron) |
| **Est. total (8 × 5 layer)** | **~900 cyc** | **~865 cyc** |

**What the compiler does better**
1. **Inlining.** There is no `BL`, no neuron PUSH/POP, and clamp_min/clamp_max stay in registers for the whole layer.
2. **One `LDMIA`** fetches output_size, clamp_min and clamp_max in the prologue.
3. **x2 unrolled inner loop** (see section 1).

**What my version does better**
1. **No spills.** After inlining, the compiler has no free registers left. Every neuron reloads `weights`, `input`, `bias`, `output`, `n-1` and `output_size` from the stack, and stores `weights_o` and `input` back. That is about 30 cycles per neuron of memory traffic, which roughly cancels what inlining saved.
2. **Strength reduction.** I advance `weights_o` with `ADD R6, R6, R9, LSL #1`, where the compiler does a `MUL` for every neuron.
3. **Down-counter with `SUBS`/`BNE`.** The compiler pays for the `uint16_t o` semantics (`BIC #0x10000`) and for re-reading `output_size`.
4. **No redundant shifts.** The compiler does `LSL #16`/`ASR #16` for `(int16_t)` and then `LSL #16`/`LSR #16` again for `(uint16_t)`.

The two end up close (my version is about 4% better by this estimate). The compiler saves the call overhead and then spends it on spills.

### 2.5 Recommendations for `dense_layer_q12_ARM`

**A. Small changes that keep the `BL neuron_q12_ARM` call** (useful if the assignment requires the layer to call the neuron):

| Now | Better | Saves |
|---|---|---|
| `LDRSH R3,[R7]` … `ADD R7,R7,#2` | `LDRSH R3, [R7], #2` | 1 instr/neuron |
| `STRH R0,[R8]` … `ADD R8,R8,#2` | `STRH R0, [R8], #2` | 1 instr/neuron |
| `MOV R0,R0,LSL#16` / `MOV R0,R0,LSR#16` / `ADD R4,R4,R0` | `MOV R0,R0,LSL #16` / `ADD R4,R4,R0,LSR #16` (shift folded into the ADD's barrel shifter) | 1 instr/neuron |
| `LDR`,`LDR`,`LDRH`,`LDRH` for the 4 stack args | `ADD r12, fp, #8` / `LDMIA r12, {…}` (AAPCS: stacked args are already extended to words) | ~5 cyc once |
| `PUSH {fp,lr}` + `MOV fp,sp` + `PUSH {v1-v7}` | one `PUSH {r4-r11, lr}`, addressing args from SP | ~3 cyc once |
| `POP {…, PC}` | `POP {…, lr}` + `BX lr` (interworking-safe on ARMv4T) | (correctness) |

Together with the optimized neuron from 1.5, this comes to about **35 + 5 × (19 + 117) ≈ 715 cycles**.

**B. Inline the neuron, but without the compiler's spills.** The work fits exactly in the 14 available registers (r0-r12 and lr). A useful side effect: the weights pointer post-increments through the row, so when the inner loop ends it already points at the next row. No `MUL` and no `ADD` are needed to get `weights_o`.

| Reg | Use | Reg | Use |
|---|---|---|---|
| r0 | input (base) | r7 | clamp_min |
| r1 | weights (walks through all rows) | r8 | clamp_max |
| r2 | bias ptr | r9 | inner counter |
| r3 | output ptr | r10 | acc / y |
| r4 | input_size | r11 | input walk ptr |
| r5 | output counter | r12, lr | load temps |
| r6 | checksum | | |

```asm
; r0=*input r1=*weights r2=*bias r3=*output
; stack: input_size, output_size, clamp_min, clamp_max
dense_layer_q12_ARM
            PUSH    {r4-r11, lr}            ; 36 bytes; leaf function, no calls -> alignment does not matter
            ADD     r12, sp, #36
            LDMIA   r12, {r4, r5, r7, r8}   ; input_size, output_size, clamp_min, clamp_max
            MOV     r6, #0                  ; checksum = 0
            CMP     r5, #0
            BEQ     d_end

d_loop      LDRSH   r10, [r2], #2           ; bias[o], bias++
            MOV     r10, r10, LSL #12       ; acc = bias << 12
            MOV     r11, r0                 ; restart input pointer
            MOVS    r9, r4, LSR #1          ; pairs = n/2 ; C = n odd ; Z = no pairs
            BCC     n_pairs
            LDRSH   r12, [r11], #2          ; odd n: peel one element
            LDRSH   lr,  [r1], #2
            MLA     r10, r12, lr, r10
n_pairs     BEQ     n_done
n_loop      LDRSH   r12, [r11], #2
            LDRSH   lr,  [r1], #2           ; r1 runs into the next row by itself
            MLA     r10, r12, lr, r10
            LDRSH   r12, [r11], #2
            LDRSH   lr,  [r1], #2
            MLA     r10, r12, lr, r10
            SUBS    r9, r9, #1
            BNE     n_loop
n_done      MOV     r10, r10, ASR #12       ; y = acc >> 12
            CMP     r10, r7
            MOVLT   r10, r7
            CMP     r10, r8
            MOVGT   r10, r8
            STRH    r10, [r3], #2           ; output[o] = y ; output++
            MOV     r12, r10, LSL #16
            ADD     r6, r6, r6, LSL #5      ; checksum *= 33
            ADD     r6, r6, r12, LSR #16    ; checksum += (uint16_t)y
            SUBS    r5, r5, #1
            BNE     d_loop

d_end       MOV     r0, r6
            POP     {r4-r11, lr}
            BX      lr
```

Estimate: ~118 cycles per neuron, **~625 cycles for the 8 × 5 layer**. That is about **30% faster** than both the current version (~865) and the compiler (~900), in 37 instructions.

### 2.6 Summary

| Version | Instructions | Est. cycles (8×5) |
|---|---|---|
| `dense_layer_q12_C` (O3-Otime, neuron inlined) | 80 | ~900 |
| `dense_layer_q12_ARM` + `neuron_q12_ARM` (current) | 53 | ~865 |
| Option A (keep `BL`, small fixes + neuron from 1.5) | ~55 | ~715 |
| Option B (inlined, unrolled x2, no spills) | 37 | ~625 |

Takeaway: on ARM7TDMI with such small sizes, `-O3 -Otime` gains little over straightforward hand-written assembly. Its generic transformations (software pipelining, two accumulators, inlining) cost registers. On this core that means extra pushes or spills, and those cancel most of the gain. The real wins come from cutting fixed overhead (prologues, calls, reloads) and from moderate unrolling.

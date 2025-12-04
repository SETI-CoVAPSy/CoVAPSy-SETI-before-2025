	.arch armv8-a
	.file	"divider.c"
	.text
	.global	set1
	.bss
	.align	3
	.type	set1, %object
	.size	set1, 40000000
set1:
	.zero	40000000
	.global	set2
	.align	3
	.type	set2, %object
	.size	set2, 40000000
set2:
	.zero	40000000
	.global	result
	.align	3
	.type	result, %object
	.size	result, 40000000
result:
	.zero	40000000
	.global	start
	.align	3
	.type	start, %object
	.size	start, 16
start:
	.zero	16
	.global	end
	.align	3
	.type	end, %object
	.size	end, 16
end:
	.zero	16
	.global	start_filing
	.align	3
	.type	start_filing, %object
	.size	start_filing, 16
start_filing:
	.zero	16
	.global	end_filing
	.align	3
	.type	end_filing, %object
	.size	end_filing, 16
end_filing:
	.zero	16
	.global	duration
	.align	3
	.type	duration, %object
	.size	duration, 8
duration:
	.zero	8
	.section	.rodata
	.align	3
.LC0:
	.string	"Filling C array"
	.align	3
.LC1:
	.string	"Fommed %d array in %ld\n"
	.align	3
.LC2:
	.string	"Division results:"
	.align	3
.LC3:
	.string	"result[%d] = %d\n"
	.align	3
.LC4:
	.string	"Time taken for division: %ld nanoseconds\n"
	.align	3
.LC5:
	.string	"Division per second in C: %f\n"
	.text
	.align	2
	.global	main
	.type	main, %function
main:
.LFB6:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	bl	puts
	adrp	x0, start_filing
	add	x1, x0, :lo12:start_filing
	mov	w0, 1
	bl	clock_gettime
	str	wzr, [sp, 20]
	b	.L2
.L3:
	bl	rand
	cmp	w0, 0
	csneg	w1, w0, w0, ge
	mov	w0, 34079
	movk	w0, 0x51eb, lsl 16
	smull	x0, w1, w0
	lsr	x0, x0, 32
	asr	w2, w0, 6
	asr	w0, w1, 31
	sub	w0, w2, w0
	mov	w2, 200
	mul	w0, w0, w2
	sub	w0, w1, w0
	add	w2, w0, 1
	adrp	x0, set2
	add	x0, x0, :lo12:set2
	ldrsw	x1, [sp, 20]
	str	w2, [x0, x1, lsl 2]
	ldr	w1, [sp, 20]
	mov	w0, 3884
	movk	w0, 0x1, lsl 16
	add	w1, w1, w0
	adrp	x0, set2
	add	x0, x0, :lo12:set2
	ldrsw	x2, [sp, 20]
	ldr	w0, [x0, x2, lsl 2]
	mul	w2, w1, w0
	adrp	x0, set1
	add	x0, x0, :lo12:set1
	ldrsw	x1, [sp, 20]
	str	w2, [x0, x1, lsl 2]
	ldr	w0, [sp, 20]
	add	w0, w0, 1
	str	w0, [sp, 20]
.L2:
	ldr	w1, [sp, 20]
	mov	w0, 38527
	movk	w0, 0x98, lsl 16
	cmp	w1, w0
	ble	.L3
	adrp	x0, end_filing
	add	x1, x0, :lo12:end_filing
	mov	w0, 1
	bl	clock_gettime
	adrp	x0, end_filing
	add	x0, x0, :lo12:end_filing
	ldr	x1, [x0]
	adrp	x0, start_filing
	add	x0, x0, :lo12:start_filing
	ldr	x0, [x0]
	sub	x0, x1, x0
	fmov	d0, x0
	scvtf	d0, d0
	mov	x0, 225833675390976
	movk	x0, 0x41cd, lsl 48
	fmov	d1, x0
	fmul	d1, d0, d1
	adrp	x0, end
	add	x0, x0, :lo12:end
	ldr	x1, [x0, 8]
	adrp	x0, start_filing
	add	x0, x0, :lo12:start_filing
	ldr	x0, [x0, 8]
	sub	x0, x1, x0
	fmov	d0, x0
	scvtf	d0, d0
	fadd	d0, d1, d0
	fcvtzs	d0, d0
	adrp	x0, duration
	add	x0, x0, :lo12:duration
	str	d0, [x0]
	adrp	x0, duration
	add	x0, x0, :lo12:duration
	ldr	x0, [x0]
	mov	x2, x0
	mov	w1, 38528
	movk	w1, 0x98, lsl 16
	adrp	x0, .LC1
	add	x0, x0, :lo12:.LC1
	bl	printf
	adrp	x0, start
	add	x1, x0, :lo12:start
	mov	w0, 1
	bl	clock_gettime
	str	wzr, [sp, 24]
	b	.L4
.L5:
	adrp	x0, set1
	add	x0, x0, :lo12:set1
	ldrsw	x1, [sp, 24]
	ldr	w1, [x0, x1, lsl 2]
	adrp	x0, set2
	add	x0, x0, :lo12:set2
	ldrsw	x2, [sp, 24]
	ldr	w0, [x0, x2, lsl 2]
	sdiv	w2, w1, w0
	adrp	x0, result
	add	x0, x0, :lo12:result
	ldrsw	x1, [sp, 24]
	str	w2, [x0, x1, lsl 2]
	ldr	w0, [sp, 24]
	add	w0, w0, 1
	str	w0, [sp, 24]
.L4:
	ldr	w1, [sp, 24]
	mov	w0, 38527
	movk	w0, 0x98, lsl 16
	cmp	w1, w0
	ble	.L5
	adrp	x0, end
	add	x1, x0, :lo12:end
	mov	w0, 1
	bl	clock_gettime
	adrp	x0, end
	add	x0, x0, :lo12:end
	ldr	x1, [x0]
	adrp	x0, start
	add	x0, x0, :lo12:start
	ldr	x0, [x0]
	sub	x0, x1, x0
	fmov	d0, x0
	scvtf	d0, d0
	mov	x0, 225833675390976
	movk	x0, 0x41cd, lsl 48
	fmov	d1, x0
	fmul	d1, d0, d1
	adrp	x0, end
	add	x0, x0, :lo12:end
	ldr	x1, [x0, 8]
	adrp	x0, start
	add	x0, x0, :lo12:start
	ldr	x0, [x0, 8]
	sub	x0, x1, x0
	fmov	d0, x0
	scvtf	d0, d0
	fadd	d0, d1, d0
	fcvtzs	d0, d0
	adrp	x0, duration
	add	x0, x0, :lo12:duration
	str	d0, [x0]
	adrp	x0, .LC2
	add	x0, x0, :lo12:.LC2
	bl	puts
	str	wzr, [sp, 28]
	b	.L6
.L7:
	adrp	x0, result
	add	x0, x0, :lo12:result
	ldrsw	x1, [sp, 28]
	ldr	w0, [x0, x1, lsl 2]
	mov	w2, w0
	ldr	w1, [sp, 28]
	adrp	x0, .LC3
	add	x0, x0, :lo12:.LC3
	bl	printf
	ldr	w0, [sp, 28]
	add	w0, w0, 1
	str	w0, [sp, 28]
.L6:
	ldr	w0, [sp, 28]
	cmp	w0, 2
	ble	.L7
	adrp	x0, duration
	add	x0, x0, :lo12:duration
	ldr	x0, [x0]
	mov	x1, x0
	adrp	x0, .LC4
	add	x0, x0, :lo12:.LC4
	bl	printf
	adrp	x0, duration
	add	x0, x0, :lo12:duration
	ldr	x0, [x0]
	scvtf	s0, x0
	fcvt	d0, s0
	mov	x0, 225833675390976
	movk	x0, 0x41cd, lsl 48
	fmov	d1, x0
	fdiv	d0, d0, d1
	mov	x0, 20684562497536
	movk	x0, 0x4163, lsl 48
	fmov	d1, x0
	fdiv	d0, d1, d0
	adrp	x0, .LC5
	add	x0, x0, :lo12:.LC5
	bl	printf
	mov	w0, 0
	ldp	x29, x30, [sp], 32
	.cfi_restore 30
	.cfi_restore 29
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits

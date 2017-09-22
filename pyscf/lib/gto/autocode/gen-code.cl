;;;;
;;;; Copyright (C) 2015-  Qiming Sun <osirpt.sun@gmail.com>
;;;; Description:
;;;;

(load "utility.cl")
(load "parser.cl")
(load "derivator.cl")

(defun gen-subscript (cells-streamer raw-script)
  (labels ((gen-tex-iter (raw-script)
             (cond ((null raw-script) raw-script)
                   ((vector? raw-script)
                    (gen-tex-iter (comp-x raw-script))
                    (gen-tex-iter (comp-y raw-script))
                    (gen-tex-iter (comp-z raw-script)))
                   ((cells? raw-script)
                    (funcall cells-streamer raw-script))
                   (t (mapcar cells-streamer raw-script)))))
    (gen-tex-iter raw-script)))

(defun convert-from-n-sys (ls n)
  (reduce (lambda (x y) (+ (* x n) y)) ls
          :initial-value 0))

(defun xyz-to-ternary (xyzs)
  (cond ((eql xyzs 'x) 0)
        ((eql xyzs 'y) 1)
        ((eql xyzs 'z) 2)
        (t (error " unknown subscript ~a" xyzs))))

(defun ternary-subscript (ops)
  "convert the polynomial xyz to the ternary"
  (cond ((null ops) ops)
        (t (convert-from-n-sys (mapcar #'xyz-to-ternary 
                                       (remove-if (lambda (x) (eql x 's))
                                                  (scripts-of ops)))
                               3))))
(defun gen-c-block (fout fmt-gout raw-script)
  (let ((ginc -1))
    (labels ((c-filter (cell)
               (let ((fac (realpart (phase-of cell)))
                     (const@3 (ternary-subscript (consts-of cell)))
                     (op@3    (ternary-subscript (ops-of cell))))
                 (if (equal fac 1)
                   (cond ((null const@3)
                          (if (null op@3)
                            (format fout " + s\[n\]" )
                            (format fout " + s\[~a*SIMDD+n\]" op@3)))
                         ((null op@3)
                          (format fout " + c\[~a\]*s\[n\]" const@3))
                         (t (format fout " + c\[~a\]*s\[~a*SIMDD+n\]" const@3 op@3)))
                   (cond ((null const@3)
                          (if (null op@3)
                            (format fout " + (~a*s\[n\])" fac)
                            (format fout " + (~a*s\[~a*SIMDD+n\])"
                                    fac op@3)))
                         ((null op@3)
                          (format fout " + (~a*c\[~a\]*s\[n\])"
                                  fac const@3))
                         (t (format fout " + (~a*c\[~a\]*s\[~a*SIMDD+n\])"
                                    fac const@3 op@3))))))
             (c-streamer (cs)
               (format fout fmt-gout (incf ginc))
               (cond ((null cs) (format fout " 0"))
                     ((cell? cs) (c-filter cs))
                     (t (mapcar #'c-filter cs)))
               (format fout ";~%")))
      (gen-subscript #'c-streamer raw-script)
      (1+ ginc))))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; effective keys are p,r,ri,...
(defun effect-keys (ops)
  (remove-if-not (lambda (x) (member x *intvar*))
                 ops))
(defun g?e-of (key)
  (case key
    ((p ip nabla px py pz) "D_")
    ((r x y z ri xi yi zi) "R_") ; the vector origin is on the center of the basis it acts on
    ((r0 x0 y0 z0 g) "R0") ; R0 ~ the vector origin is (0,0,0)
    ((rc xc yc zc) "RC") ; the vector origin is set in env[PTR_COMMON_ORIG]
    ((nabla-rinv nabla-r12 breit-r1 breit-r2) "D_")
    (otherwise (error "unknown key ~a~%" key))))

(defun dump-header (fout)
  (format fout "/*
 * Copyright (C) 2016-  Qiming Sun <osirpt.sun@gmail.com>
 * Description: code generated by  gen-code.cl
 */
#include \"grid_ao_drv.h\"
"))

(defun dump-declare-dri-for-rc (fout i-ops symb)
  (when (intersection '(rc xc yc zc) i-ops)
    (format fout "double dr~a[3];~%" symb)
    (format fout "dr~a[0] = r~a[0] - env[PTR_COMMON_ORIG+0];~%" symb symb)
    (format fout "dr~a[1] = r~a[1] - env[PTR_COMMON_ORIG+1];~%" symb symb)
    (format fout "dr~a[2] = r~a[2] - env[PTR_COMMON_ORIG+2];~%" symb symb))
  (when (intersection '(ri xi yi zi) i-ops)
    (if (intersection '(rc xc yc zc) i-ops)
      (error "Cannot declare dri because rc and ri coexist"))
    (format fout "double dr~a[3];~%" symb)
    (format fout "dr~a[0] = r~a[0] - ri[0];~%" symb symb)
    (format fout "dr~a[1] = r~a[1] - ri[1];~%" symb symb)
    (format fout "dr~a[2] = r~a[2] - ri[2];~%" symb symb)))

(defun dump-declare-giao (fout expr)
  (let ((n-giao (count 'g expr)))
    (when (> n-giao 0)
      (format fout "double c[~a];~%" (expt 3 n-giao))
      (loop
        for i upto (1- (expt 3 n-giao)) do
        (format fout "c[~a] = 1" i)
        (loop
          for j from (1- n-giao) downto 0
          and res = i then (multiple-value-bind (int res) (floor res (expt 3 j))
                             (format fout " * (-ri[~a])" int)
                             res))
        (format fout ";~%")))))

(defun last-bit1 (n)
  ; how many 0s follow the last bit 1
  (loop
    for i upto 31
    thereis (if (oddp (ash n (- i))) i)))
(defun combo-op (fout fmt-op op-rev ig)
  (let* ((right (last-bit1 ig))
         (ig0 (- ig (ash 1 right)))
         (op (nth right op-rev)))
    (format fout fmt-op (g?e-of op) ig ig0 right)))

(defun power2-range (n &optional (shift 0))
  (range (+ shift (ash 1 n)) (+ shift (ash 1 (1+ n)))))
(defun dump-combo-op (fout fmt-op op-rev)
  (let ((op-len (length op-rev)))
    (loop
      for right from 0 to (1- op-len) do
      (loop
        for ig in (power2-range right) do
        (combo-op fout fmt-op op-rev ig)))))

(defun dec-to-ybin (n)
  (parse-integer (substitute #\0 #\2 (write-to-string n :base 3))
                 :radix 2))
(defun dec-to-zbin (n)
  (parse-integer (substitute #\1 #\2
                             (substitute #\0 #\1
                                         (write-to-string n :base 3)))
                 :radix 2))
(defun dump-s-1e (fout n)
  (loop
    for i upto (1- (expt 3 n)) do
    (let* ((ybin (dec-to-ybin i))
           (zbin (dec-to-zbin i))
           (xbin (- (ash 1 n) 1 ybin zbin)))
      (format fout "s[~a*SIMDD+n] = e * fx~a[lx*SIMDD+n] * fy~a[ly*SIMDD+n] * fz~a[lz*SIMDD+n];~%"
              i xbin ybin zbin))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun gen-code-eval-ao (fout intname expr)
  (let* ((op-rev (reverse (effect-keys expr)))
         (op-len (length op-rev))
         (raw-script (eval-gto expr))
         (ts1 (car raw-script))
         (sf1 (cadr raw-script))
         (strout (make-string-output-stream))
         (goutinc (gen-c-block strout "buf[~d*SIMDD+n] =" (last1 raw-script)))
         (codestr (get-output-stream-string strout))
         (e1comps (if (eql sf1 'sf) 1 4))
         (tensors (/ goutinc e1comps)))
    (format fout "/*  ~{~a ~}|GTO> */~%" expr)
    (format fout "static void shell_eval_~a(double *cgto, double *ri, double *exps,
double *coord, double *alpha, double *coeff, double *env,
int l, int np, int nc, int nao, int ngrids, int bgrids)
{" intname)
    (format fout "
const int degen = (l+1)*(l+2)/2;
int lx, ly, lz, i, j, j1, k, l1, n;
double e;
double *pgto;
double *gridx = coord;
double *gridy = coord+BLKSIZE;
double *gridz = coord+BLKSIZE*2;
double fx0[SIMDD*16*~a];
double fy0[SIMDD*16*~a];
double fz0[SIMDD*16*~a];~%" (ash 1 op-len) (ash 1 op-len) (ash 1 op-len))
    (loop
       for i in (range (1- (ash 1 op-len))) do
         (format fout "double *fx~d = fx~d + SIMDD*16;~%" (1+ i) i)
         (format fout "double *fy~d = fy~d + SIMDD*16;~%" (1+ i) i)
         (format fout "double *fz~d = fz~d + SIMDD*16;~%" (1+ i) i))
    (format fout "double buf[SIMDD*nc*~d];~%" goutinc)
    (format fout "double s[SIMDD*~d];~%" (expt 3 op-len))
    (format fout "double *gto0 = cgto;~%")
    (loop
       for i in (range (1- goutinc)) do
         (format fout "double *gto~d = cgto + nao*ngrids*~d;~%" (1+ i) (1+ i)))
    (dump-declare-dri-for-rc fout expr "i")
    (dump-declare-giao fout expr)

    (format fout "
for (j = 0; j < ~d; j++) {
        pgto = cgto + j*nao*ngrids;
        for (n = 0; n < degen*nc; n++) {
        for (i = 0; i < bgrids; i++) {
                pgto[n*ngrids+i] = 0;
        } }
}" goutinc)
    (format fout "
for (i = 0; i < bgrids+1-SIMDD; i+=SIMDD) {
        for (k = 0; k < np; k++) {
                if (_nonzero_in(exps+k*BLKSIZE+i, SIMDD)) {
for (n = 0; n < SIMDD; n++) {
        fx0[n] = 1;
        fy0[n] = 1;
        fz0[n] = 1;
}
for (lx = 1; lx <= l+~d; lx++) {
for (n = 0; n < SIMDD; n++) {
        fx0[lx*SIMDD+n] = fx0[(lx-1)*SIMDD+n] * gridx[i+n];
        fy0[lx*SIMDD+n] = fy0[(lx-1)*SIMDD+n] * gridy[i+n];
        fz0[lx*SIMDD+n] = fz0[(lx-1)*SIMDD+n] * gridz[i+n];
} }~%" op-len)
;;; generate g_(bin)
    (dump-combo-op fout "GTO_~aI(~d, ~d, l+~a);~%" op-rev)
;;; dump result of eval-int
    (format fout "for (lx = l, l1 = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, l1++) {
                lz = l - lx - ly;
                for (n = 0; n < SIMDD; n++) {
                       e = exps[k*BLKSIZE+i+n];~%")
    (dump-s-1e fout op-len)
    (format fout "                }
                for (n = 0; n < SIMDD; n++) {~%")
    (format fout "~a" codestr)
    (format fout "                }
                for (j = 0, j1 = l1; j < nc; j++, j1+=degen) {
                for (n = 0; n < SIMDD; n++) {~%")
    (loop
       for i in (range goutinc) do
         (format fout "gto~d[j1*ngrids+i+n] += buf[~d*SIMDD+n] * coeff[j*np+k];~%" i i))
    (format fout "} } } } } } }~%")
    (format fout "
if (i < bgrids) {
        for (k = 0; k < np; k++) {
                if (_nonzero_in(exps+k*BLKSIZE+i, bgrids-i)) {
for (n = 0; n < SIMDD; n++) {
        fx0[n] = 1;
        fy0[n] = 1;
        fz0[n] = 1;
}
for (lx = 1; lx <= l+~d; lx++) {
for (n = 0; n < SIMDD; n++) {
        fx0[lx*SIMDD+n] = fx0[(lx-1)*SIMDD+n] * gridx[i+n];
        fy0[lx*SIMDD+n] = fy0[(lx-1)*SIMDD+n] * gridy[i+n];
        fz0[lx*SIMDD+n] = fz0[(lx-1)*SIMDD+n] * gridz[i+n];
} }~%" op-len)
;;; generate g_(bin)
    (dump-combo-op fout "GTO_~aI(~d, ~d, l+~a);~%" op-rev)
;;; dump result of eval-int
    (format fout "for (lx = l, l1 = 0; lx >= 0; lx--) {
        for (ly = l - lx; ly >= 0; ly--, l1++) {
                lz = l - lx - ly;
                for (n = 0; n < SIMDD; n++) {
                       e = exps[k*BLKSIZE+i+n];~%")
    (dump-s-1e fout op-len)
    (format fout "                }
                for (n = 0; n < SIMDD; n++) {~%")
    (format fout "~a" codestr)
    (format fout "                }
                for (j = 0, j1 = l1; j < nc; j++, j1+=degen) {
                for (n = 0; n < bgrids-i; n++) {~%")
    (loop
       for i in (range goutinc) do
         (format fout "gto~d[j1*ngrids+i+n] += buf[~d*SIMDD+n] * coeff[j*np+k];~%" i i))
    (format fout "} } } } } } }~%")
    (format fout "}~%")

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; _cart
    (format fout "void ~a_cart(int ngrids, int *shls_slice, int *ao_loc,
double *ao, double *coord, char *non0table,
int *atm, int natm, int *bas, int nbas, double *env)
{~%" intname)
    (format fout "int param[] = {~d, ~d};~%" e1comps tensors)
    (format fout "GTOeval_cart_drv(shell_eval_~a, GTOprim_exp, ~a,
ngrids, param, shls_slice, ao_loc, ao, coord, non0table,
atm, natm, bas, nbas, env);~%}~%" intname (factor-of expr))
;;; _sph
    (format fout "void ~a_sph(int ngrids, int *shls_slice, int *ao_loc,
double *ao, double *coord, char *non0table,
int *atm, int natm, int *bas, int nbas, double *env)
{~%" intname)
    (format fout "int param[] = {~d, ~d};~%" e1comps tensors)
    (format fout "GTOeval_sph_drv(shell_eval_~a, GTOprim_exp, ~a,
ngrids, param, shls_slice, ao_loc, ao, coord, non0table,
atm, natm, bas, nbas, env);~%}~%" intname (factor-of expr))
;;; _spinor
    (format fout "void ~a_spinor(int ngrids, int *shls_slice, int *ao_loc,
double complex *ao, double *coord, char *non0table,
int *atm, int natm, int *bas, int nbas, double *env)
{~%" intname)
    (format fout "int param[] = {~d, ~d};~%" e1comps tensors)
    (format fout "GTOeval_spinor_drv(shell_eval_~a, GTOprim_exp, CINTc2s_~aket_spinor_~a, ~a,
ngrids, param, shls_slice, ao_loc, ao, coord, non0table, atm, natm, bas, nbas, env);~%}~%"
            intname
            (if (eql ts1 'ts) "" "i")
            (if (eql sf1 'sf) "sf1" "si1")
            (factor-of expr))))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun gen-eval (filename &rest items)
  (with-open-file (fout (mkstr filename)
                        :direction :output :if-exists :supersede)
    (dump-header fout)
    (flet ((gen-code (item)
             (let ((intname (mkstr (car item)))
                   (raw-infix (cadr item)))
               (gen-code-eval-ao fout intname raw-infix))))
      (mapcar #'gen-code items))))

;; vim: ft=lisp
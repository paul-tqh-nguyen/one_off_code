;; Less Invasive Settings

(setq inhibit-startup-screen t)
(load-theme 'manoj-dark)
(show-paren-mode 1)
(setq visible-bell 1)
(add-to-list 'display-buffer-alist '("^\\*shell\\*$" . (display-buffer-same-window)))
(when (version<= "26.0.50" emacs-version)
  (global-display-line-numbers-mode))

;; Update EMACS Load Path

(package-initialize)
(add-to-list 'load-path "/home/pnguyen/code/")

;; Undo Tree

(load "undo-tree.el")
(global-undo-tree-mode)

;; Javascript Settings

(load "rjsx-mode.el")
(load "js2-mode.el")
(add-to-list 'auto-mode-alist '("\\.js\\'" . js2-mode))
(add-to-list 'auto-mode-alist '("\\.js\\'" . rjsx-mode))
(setq sgml-attribute-offset 2) ;; @hack figure out why we have to this

;; Custom Functions

(defun escape-quotes (@begin @end)
  "Escapes quotes in a region"
  (interactive
   (if (use-region-p)
       (list (region-beginning) (region-end))
     (list (line-beginning-position) (line-end-position))))
  (save-excursion
      (save-restriction
        (narrow-to-region @begin @end)
        (goto-char (point-min))
        (while (search-forward "\"" nil t)
          (replace-match "\\\"" "FIXEDCASE" "LITERAL")))))

(defun new-shell ()
  "Creates a new shell buffer"
  ;; @todo this currently doesn't work via M-x ; make it work
  (interactive)
  (shell (generate-new-buffer-name "*shell*")))

(defun start-shell-buffer-with-name (buffer-name)
  (interactive
    (list 
      (if current-prefix-arg
          nil
          (read-from-minibuffer "New Shell Buffer Name: "))))
  (if (null (get-buffer buffer-name))
      (shell buffer-name)
    (switch-to-buffer buffer-name)))

(defun tmp ()
  (interactive)
  (add-to-list 'display-buffer-alist '("^\\*tmp\\*$" . (display-buffer-same-window)))
  (start-shell-buffer-with-name "*tmp*"))

(defun runner ()
  (interactive)
  (add-to-list 'display-buffer-alist '("^\\*runner\\*$" . (display-buffer-same-window)))
  (start-shell-buffer-with-name "*runner*"))

(defun second ()
  (interactive)
  (add-to-list 'display-buffer-alist '("^\\*second\\*$" . (display-buffer-same-window)))
  (start-shell-buffer-with-name "*second*"))

(defun thrid ()
  (interactive)
  (add-to-list 'display-buffer-alist '("^\\*thrid\\*$" . (display-buffer-same-window)))
  (start-shell-buffer-with-name "*thrid*"))

(defun fourth ()
  (interactive)
  (add-to-list 'display-buffer-alist '("^\\*fourth\\*$" . (display-buffer-same-window)))
  (start-shell-buffer-with-name "*fourth*"))

(defun fifth ()
  (interactive)
  (add-to-list 'display-buffer-alist '("^\\*fifth\\*$" . (display-buffer-same-window)))
  (start-shell-buffer-with-name "*fifth*"))

(defun python ()
  (interactive)
  (add-to-list 'display-buffer-alist '("^\\*python\\*$" . (display-buffer-same-window)))
  (start-shell-buffer-with-name "*python*"))

(defun node ()
  (interactive)
  (add-to-list 'display-buffer-alist '("^\\*node\\*$" . (display-buffer-same-window)))
  (start-shell-buffer-with-name "*node*"))

(defun server ()
  (interactive)
  (add-to-list 'display-buffer-alist '("^\\*server\\*$" . (display-buffer-same-window)))
  (start-shell-buffer-with-name "*server*"))

;; Shortcut Keys

(global-set-key (kbd "C-c \"") 'escape-quotes)
(global-set-key (kbd "C-x C-b") 'buffer-menu)
(global-set-key (kbd "C-x a") 'undo-tree-visualize)
(global-set-key (kbd "C-c ;") 'comment-region)
(global-set-key (kbd "C-c :") 'uncomment-region)
;;(global-set-key (kbd "C-M-<left>") 'backward-sexp) ;; @todo get this working
;;(global-set-key (kbd "C-M-<right>") 'forward-sexp) ;; @todo get this working
(put 'downcase-region 'disabled nil)
(put 'upcase-region 'disabled nil)

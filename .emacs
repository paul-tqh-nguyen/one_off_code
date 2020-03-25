;; Less Invasive Settings

(setq inhibit-startup-screen t)
(load-theme 'manoj-dark)
(show-paren-mode 1)
(setq visible-bell 1)
(add-to-list 'display-buffer-alist '("^\\*shell\\*$" . (display-buffer-same-window)))
(when (version<= "26.0.50" emacs-version)
  (global-display-line-numbers-mode))
(setq tramp-default-method "ssh")
(add-to-list 'display-buffer-alist (cons "\\*Async Shell Command\\*.*" (cons #'display-buffer-no-window nil)))

;; Keep Settings updated

(shell-command-to-string "pushd ~/code/one_off_code/ ; git pull ; popd")

;; Misc. OS-Specific / Machine-Specific Changes

(cond
 ((eq system-type 'darwin)
  (setq mac-command-modifier 'meta)
  (setq mac-function-modifier 'control)
  (setq mac-option-modifier nil))
 ((eq system-type 'gnu/linux)
  )
 (t
  (format "Could not determine OS flavor.")))

;; Update EMACS Load Path

(package-initialize)
(let* ((currently-executing-file-name user-init-file)
       (expanded-file-name (expand-file-name currently-executing-file-name))
       (absolute-file-name (file-chase-links expanded-file-name))
       (absolute-file-name-directory (file-name-directory absolute-file-name))
       (emacs-files-directory (format "%s%s" absolute-file-name-directory "emacs_files/")))
  (add-to-list 'load-path emacs-files-directory))

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

(defmacro create-named-shell-function (function-name)
  (let ((buffer-name-regex-string (format "^\\*%s\\*$" function-name))
	(buffer-name (format "*%s*" function-name)))
    `(defun ,function-name ()
       (interactive)
       (add-to-list 'display-buffer-alist '(,buffer-name-regex-string . (display-buffer-same-window)))
       (start-shell-buffer-with-name ,buffer-name))))

(defmacro create-named-shell-functions (&rest function-names)
  (let (commands)
    (dolist (function-name function-names)
      (push `(create-named-shell-function ,function-name) commands))
    (setq commands (nreverse commands))
    `(list ,@commands)))

(create-named-shell-functions
 
 python
 node
 server
 
 tmp
 second
 third
 fourth
 fifth
 )

(defun start-remote-ssh-shell-buffer-with-name (username host buffer-name &optional async-command-strings)
  (if (get-buffer buffer-name)
      (switch-to-buffer buffer-name)
    (let ((default-directory (format "/ssh:%s@%s:" username host)))
      (mapcar #'async-shell-command async-command-strings)
      (add-to-list 'display-buffer-alist `(,buffer-name . (display-buffer-same-window)))
      (start-shell-buffer-with-name buffer-name))))

(defmacro create-named-cuda-shell-function (function-name)
  (let ((buffer-name (format "*%s*" function-name)))
    `(defun ,function-name ()
       (interactive)
       (let* ((username "pnguyen")
	      (host "192.168.131.229")
	      (buffer-name ,buffer-name)
	      (async-command-strings '("(cd /home/pnguyen/code/one_off_code/ ; git pull)"))
	      (default-directory (format "/ssh:%s@%s:" username host)))
	 (start-remote-ssh-shell-buffer-with-name username host buffer-name async-command-strings)))))

(defmacro create-named-cuda-shell-functions (&rest function-names)
  (let (commands)
    (dolist (function-name function-names)
      (push `(create-named-cuda-shell-function ,function-name) commands))
    (setq commands (nreverse commands))
    `(list ,@commands)))

(create-named-cuda-shell-functions
 cuda
 cuda-python
 cuda-shell
 )

;; shortcut keys

(global-set-key (kbd "C-c \"") 'escape-quotes)
(global-set-key (kbd "C-x C-b") 'buffer-menu)
(global-set-key (kbd "C-x a") 'undo-tree-visualize)
(global-set-key (kbd "C-c ;") 'comment-region)
(global-set-key (kbd "C-c :") 'uncomment-region)
(global-set-key (kbd "<C-down>") 'forward-paragraph)
(global-set-key (kbd "<C-up>") 'backward-paragraph)
(global-set-key (kbd "<C-M-left>") 'backward-sexp) ;; @todo get this working
(global-set-key (kbd "<C-M-right>") 'forward-sexp) ;; @todo get this working
(put 'downcase-region 'disabled nil)
(put 'upcase-region 'disabled nil)
(put 'dired-find-alternate-file 'disabled nil)

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
(add-to-list 'display-buffer-alist (cons "\\*Shell Command Output\\*.*" (cons #'display-buffer-no-window nil)))

;; Update EMACS Load Path

(package-initialize)
(let* ((currently-executing-file-name user-init-file)
       (expanded-file-name (expand-file-name currently-executing-file-name))
       (absolute-file-name (file-chase-links expanded-file-name))
       (absolute-file-name-directory (file-name-directory absolute-file-name))
       (emacs-files-directory (format "%s%s" absolute-file-name-directory "emacs_files/")))
  (add-to-list 'load-path emacs-files-directory))

;; GUI Settings

(tool-bar-mode -1)
(menu-bar-mode -1)
(set-frame-font "Consolas" nil t)

;; TRAMP Term

(load "tramp-term.el")
(defun cuda-tramp-term () 
  (interactive) 
  (tramp-term '("pnguyen" "colo-dgx-01.corp.continuum.io")))

;; Framemove

(load "framemove.el")

;; Undo Tree

(load "undo-tree.el")
(global-undo-tree-mode)

;; YAML Mode

(load "yaml-mode.el")

;; LLVM Settings

(load "tablegen-mode.el")
(load "llvm-mode.el")
(load "mlir-mode.el")

;; Cython Mode

(load "cython-mode.el")

;; Javascript Settings

(load "json-reformat.el")
(load "json-snatcher.el")
(load "json-mode.el")
(add-to-list 'auto-mode-alist '("\\.json\\'" . json-mode))
(add-to-list 'auto-mode-alist '("\\.geojson\\'" . json-mode))

(load "js2-mode.el")
(add-to-list 'auto-mode-alist '("\\.js\\'" . js2-mode))

(load "rjsx-mode.el")
(setq sgml-attribute-offset 2) ;; @hack figure out why we have to this
(add-to-list 'auto-mode-alist '("\\.js\\'" . rjsx-mode))

(defun javascript-printf-selected ()
  (interactive)
  (when (use-region-p)
    (let* ((region-string (buffer-substring (region-beginning) (region-end)))
	   (region-lines (split-string region-string "\n"))
	   (number-of-region-lines (length region-lines))
	   (current-line-index 0))
      (delete-region (region-beginning) (region-end))
      (dolist (region-line-unnormalized region-lines)
	(let* ((region-line (replace-regexp-in-string "	" "        " region-line-unnormalized))
	       (index-of-first-non-white-space-character (string-match "[^\s-]" region-line))
	       (no-indentation-region-line (if index-of-first-non-white-space-character (substring region-line index-of-first-non-white-space-character nil) region-line)))
	  (if (null index-of-first-non-white-space-character)
	      (insert region-line)
	    (progn
	      (dotimes (space-index index-of-first-non-white-space-character)
	  	(insert " "))
	      (insert (format "console.log(`%s ${JSON.stringify(%s)}`);" no-indentation-region-line no-indentation-region-line))))
	  (unless (eq current-line-index (1- number-of-region-lines))
	    (insert "\n")
	    (setq current-line-index (1+ current-line-index))))))))

(add-hook 'js2-mode-hook
	  (lambda ()
	    (local-set-key (kbd "C-c p") 'javascript-printf-selected)))

(add-hook 'js-mode-hook
	  (lambda ()
	    (local-set-key (kbd "C-c p") 'javascript-printf-selected)))

(add-hook 'rjsx-mode-hook
	  (lambda ()
	    (local-set-key (kbd "C-c p") 'javascript-printf-selected)))

(add-hook 'html-mode-hook
	  (lambda ()
	    (local-set-key (kbd "C-M-q") 'indent-region)))

;; Custom Functions

(defvar *dev-dir* "c:\\Users\\trslcj\\code\\MSG-Quant")

(defun set-dev-dir (new-dev-dir)
  (interactive "DNew Dev Dir: ")
  (setq *dev-dir* new-dev-dir))

(defun grep-os-specific (grep-bash-command)
  (cond
   ((eq system-type 'windows-nt)
    (let ((dwimmed-grep-bash-command (format "set \"PATH=C:\\Users\\trslcj\\AppData\\Local\\miniconda3\\Library\\bin\" && %s && REM " grep-bash-command))
	  (default-directory *dev-dir*))
      (grep dwimmed-grep-bash-command)))
   (t
    (grep grep-bash-command))))

(defun general-metapoint (search-string)
  (interactive "M(General Metapoint) Search Text: ")
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  (progn ;; visit file in same window ; stolen from https://emacs.stackexchange.com/questions/33857/open-search-result-in-the-same-window
    (defun my-compile-goto-error-same-window ()
      (interactive)
      (let ((display-buffer-overriding-action '((display-buffer-reuse-window display-buffer-same-window)
						(inhibit-same-window . nil))))
	(call-interactively #'compile-goto-error)))
    (defun my-compilation-mode-hook ()
      (local-set-key (kbd "SPC") #'my-compile-goto-error-same-window))
    (add-hook 'compilation-mode-hook #'my-compilation-mode-hook))
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  (let ((grep-bash-command (format "cd %s && echo && git grep -IrFn \"%s\"" *dev-dir* search-string)))
    (grep-os-specific grep-bash-command)))

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

(defun camel-case-to-snake-case () 
  (interactive) 
  (replace-regexp "\\([A-Z]\\)" "_\\1" nil (region-beginning) (region-end))
  (downcase-region (region-beginning) (region-end)))

(defun htop ()
  (interactive)
  (let* ((buffer-basename "htop")
	 (buffer-name (format "*%s*" buffer-basename)))
    (if (get-buffer buffer-name)
	(switch-to-buffer buffer-name)
      (let* ((bash-executable (shell-command-to-string "which bash | xargs echo -n")))
	(ansi-term bash-executable buffer-basename)
        (display-line-numbers-mode -1)
	(term-send-raw-string "htop -d1 ; exit \n")
	))))

(advice-add ;; for htop
 'term-handle-exit
 :after (lambda (&optional process-name msg)
	  (when (string= process-name "*htop*")
	    (kill-buffer process-name))))

(defun scratch ()
   "Create a scratch buffer"
   (interactive)
   (switch-to-buffer (get-buffer-create "*scratch*"))
   ;;(find-file "~/scratch/scratch.txt")
   )

(defun new-shell ()
  "Creates a new shell buffer"
  ;; @todo this currently doesn't work via M-x ; make it work
  (interactive)
  (shell (generate-new-buffer-name "*shell*")))

(defun start-shell-buffer-with-name (buffer-name init-command)
  (if (or (null (get-buffer buffer-name))
	  (null (get-buffer-process buffer-name)))
      (progn 
	(shell buffer-name)
	(end-of-buffer)
	(let ((start-point (point))
	      (buffer-process (get-buffer-process buffer-name))
	      (shell-dirstack-query "pwd"))
	  (insert init-command)
	  (comint-send-input)
	  (accept-process-output buffer-process)
	  (delete-region start-point (buffer-size))
	  (shell-resync-dirs)
	  (end-of-buffer)))
    (switch-to-buffer buffer-name)))

(defmacro create-named-shell-function (function-name)
  (let ((buffer-name-regex-string (format "^\\*%s\\*$" function-name))
	(buffer-name (format "*%s*" function-name)))
    `(defun ,function-name ()
       (interactive)
       (add-to-list 'display-buffer-alist '(,buffer-name-regex-string . (display-buffer-same-window)))
       (start-shell-buffer-with-name ,buffer-name "echo"))))

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

 ssh-tunnel
 )
(fmakunbound 'secrets-show-secrets) ;; don't have a similar name to "second"

(defun start-remote-ssh-shell-buffer-with-name (username host buffer-name shell-start-up-command)
  (let ((default-directory (format "/ssh:%s@%s:" username host)))
    (add-to-list 'display-buffer-alist `(,buffer-name . (display-buffer-same-window)))
    (start-shell-buffer-with-name buffer-name shell-start-up-command)))

;; Start Up Shells

(mapcar #'funcall '(
		    shell
		    python
 
		    tmp
		    second
		    third
		    fourth
		    fifth

		    ssh-tunnel
		    ielm
		    
		    shell))

(defun broadcast-local (broadcast-command)
  (interactive "MCommand to broadcast: ")
  (let ((shell-loaders '(
			 shell
			 python
 
			 tmp
			 second
			 third
			 fourth
			 fifth

			 ssh-tunnel
			 )))
    (dolist (shell-loader shell-loaders)
      (funcall shell-loader)
      (end-of-buffer)
      (insert broadcast-command)
      (comint-send-input)
      )
    (shell)))

;; Shortcut Keys

(global-set-key (kbd "C-c \"") 'escape-quotes)
(global-set-key (kbd "C-x C-b") 'buffer-menu)
(global-set-key (kbd "C-x a") 'undo-tree-visualize)
(global-set-key (kbd "C-c `") 'camel-case-to-snake-case)
(global-set-key (kbd "C-c ;") 'comment-region)
(global-set-key (kbd "C-c :") 'uncomment-region)
(global-set-key (kbd "<M-down>") 'forward-paragraph)
(global-set-key (kbd "<M-up>") 'backward-paragraph)
(global-set-key (kbd "M-,") 'general-metapoint)

(global-set-key (kbd "C-c <up>") 'windmove-up)
(global-set-key (kbd "C-c <down>") 'windmove-down)
(global-set-key (kbd "C-c <left>") 'windmove-left)
(global-set-key (kbd "C-c <right>") 'windmove-right)

(put 'downcase-region 'disabled nil)
(put 'upcase-region 'disabled nil)
(put 'dired-find-alternate-file 'disabled nil)

;; Python Settings

(defun python-printf-selected ()
  (interactive)
  (when (use-region-p)
    (let* ((region-string (buffer-substring (region-beginning) (region-end)))
	   (region-lines (split-string region-string "\n"))
	   (number-of-region-lines (length region-lines))
	   (current-line-index 0))
      (delete-region (region-beginning) (region-end))
      (dolist (region-line-unnormalized region-lines)
	(let* ((region-line (replace-regexp-in-string "	" "        " region-line-unnormalized))
	       (index-of-first-non-white-space-character (string-match "[^\s-]" region-line))
	       (no-indentation-region-line (if index-of-first-non-white-space-character (substring region-line index-of-first-non-white-space-character nil) region-line)))
	  (if (null index-of-first-non-white-space-character)
	      (insert region-line)
	    (progn
	      (dotimes (space-index index-of-first-non-white-space-character)
	  	(insert " "))
	      (insert (format "print(f\"%s {repr(%s)}\")" no-indentation-region-line no-indentation-region-line))))
	  (unless (eq current-line-index (1- number-of-region-lines))
	    (insert "\n")
	    (setq current-line-index (1+ current-line-index))))))))

(add-hook 'python-mode-hook
	  (lambda ()
	    (local-set-key (kbd "C-c p") 'python-printf-selected)))

;; C/C++ Settings

(defun cpp-printf-selected ()
  (interactive)
  (when (use-region-p)
    (let* ((region-string (buffer-substring (region-beginning) (region-end)))
	   (region-lines (split-string region-string "\n"))
	   (number-of-region-lines (length region-lines))
	   (current-line-index 0))
      (delete-region (region-beginning) (region-end))
      (dolist (region-line-unnormalized region-lines)
	(let* ((region-line (replace-regexp-in-string "	" "        " region-line-unnormalized))
	       (index-of-first-non-white-space-character (string-match "[^\s-]" region-line))
	       (no-indentation-region-line (if index-of-first-non-white-space-character (substring region-line index-of-first-non-white-space-character nil) region-line)))
	  (if (null index-of-first-non-white-space-character)
	      (insert region-line)
	    (progn
	      (dotimes (space-index index-of-first-non-white-space-character)
	  	(insert " "))
	      (insert (format "std::cout << \"%s: \" << %s << std::endl;" no-indentation-region-line no-indentation-region-line))))
	  (unless (eq current-line-index (1- number-of-region-lines))
	    (insert "\n")
	    (setq current-line-index (1+ current-line-index))))))))

(add-hook 'c++-mode-hook
	  (lambda ()
	    (local-set-key (kbd "C-c p") 'cpp-printf-selected)))

;; Misc. OS-Specific / Machine-Specific Changes

(cond
 ((eq system-type 'darwin)
  (setq ispell-program-name "/usr/local/bin/aspell")
  (setq mac-command-modifier 'meta)
  (setq mac-function-modifier 'control)
  (setq mac-option-modifier ni))
 ((eq system-type 'gnu/linux)
  )
 ((eq system-type 'windows-nt)
  (setq shell-dirstack-query "echo %cd%")
  (broadcast-local "conda activate base")
  (broadcast-local "conda activate msg"))
 (t
  (format "Could not determine OS flavor.")))

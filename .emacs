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

;; Keep Settings updated

(shell-command-to-string "pushd ~/code/one_off_code/ ; git pull ; popd")

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

(defun javascript-printf-selected ()
  (interactive)
  (when (use-region-p)
    (let* ((region-string (buffer-substring (region-beginning) (region-end)))
	   (region-lines (split-string region-string "\n"))
	   (number-of-region-lines (length region-lines))
	   (current-line-index 0))
      (delete-region (region-beginning) (region-end))
      (dolist (region-line region-lines)
	(let* ((index-of-first-non-white-space-character (string-match "[^\s-]" region-line))
	       (no-indentation-region-line (if index-of-first-non-white-space-character (substring region-line index-of-first-non-white-space-character nil) region-line)))
	  (if (null index-of-first-non-white-space-character)
	      (insert region-line)
	    (progn
	      (dotimes (space-index index-of-first-non-white-space-character)
	  	(insert " "))
	      (insert (format "console.log(`%s ${%s}`);" no-indentation-region-line no-indentation-region-line))))
	  (unless (eq current-line-index (1- number-of-region-lines))
	    (insert "\n")
	    (setq current-line-index (1+ current-line-index))))))))

(add-hook 'rjsx-mode-hook
	  (lambda ()
	    (local-set-key (kbd "C-c p") 'javascript-printf-selected)))

(add-hook 'html-mode-hook
	  (lambda ()
	    (local-set-key (kbd "C-M-q") 'indent-region)))

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

(defun camel-case-to-dashes () 
  (interactive) 
  (replace-regexp "\\([A-Z]\\)" "-\\1" nil (region-beginning) (region-end))
  (downcase-region (region-beginning) (region-end)))

(defun scratch ()
   "Create a scratch buffer"
   (interactive)
   (switch-to-buffer (get-buffer-create "*scratch*")))

(defun new-shell ()
  "Creates a new shell buffer"
  ;; @todo this currently doesn't work via M-x ; make it work
  (interactive)
  (shell (generate-new-buffer-name "*shell*")))

(defun start-shell-buffer-with-name (buffer-name init-command)
  (if (null (get-buffer buffer-name))
      (progn 
	(shell buffer-name)
	(shell-resync-dirs)
	(insert init-command)
	(comint-send-input)
	(insert "echo")
	(comint-send-input))
    (progn
      (switch-to-buffer buffer-name)
      (shell-resync-dirs))))

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

(defun start-remote-ssh-shell-buffer-with-name (username host buffer-name shell-start-up-command)
  (if (get-buffer buffer-name)
      (switch-to-buffer buffer-name)
    (let ((default-directory (format "/ssh:%s@%s:" username host)))
      (add-to-list 'display-buffer-alist `(,buffer-name . (display-buffer-same-window)))
      (start-shell-buffer-with-name buffer-name shell-start-up-command))))

(defmacro create-named-cuda-shell-function (function-name)
  (let ((buffer-name (format "*%s*" function-name)))
    `(defun ,function-name ()
       (interactive)
       (let* ((username "pnguyen")
	      (host "colo-dgx-01.corp.continuum.io")
	      (buffer-name ,buffer-name)
	      (default-directory (format "/ssh:%s@%s:" username host)))
	 (start-remote-ssh-shell-buffer-with-name username host buffer-name "conda activate mg")))))

(defmacro create-named-cuda-shell-functions (&rest function-names)
  (let (commands)
    (dolist (function-name function-names)
      (push `(create-named-cuda-shell-function ,function-name) commands))
    (setq commands (nreverse commands))
    `(list ,@commands)))

(create-named-cuda-shell-functions
 cuda
 cuda-shell
 cuda-python
 cuda-test
 cuda-second
 cuda-third
 cuda-fourth
 cuda-fifth
 jupyter
 gpu1
 gpu2
 gpu3
 gpu4
 gpu5
 gpu6
 gpu7
 gpu8
 gpu9
 )

(defun start-cuda-shells ()
  (interactive)
  (mapcar #'funcall '(
		      cuda
		      shell
		      
		      cuda-shell
		      shell
		      
		      cuda-test
		      shell
		      
		      cuda-python
		      shell
		      
		      cuda-second
		      shell
		      
		      cuda-third
		      shell
		      
		      cuda-fourth
		      shell
		      
		      cuda-fifth
		      shell
		      
		      jupyter
		      shell
		      
		      gpu1
		      shell
		      
		      gpu2
		      shell
		      
		      gpu3
		      shell
		      
		      gpu4
		      shell
		      
		      gpu5
		      shell
		      
		      gpu6
		      shell
		      
		      gpu7
		      shell
		      
		      gpu8
		      shell
		      
		      gpu9
		      shell
		      )))

(defun gpu-farm-int (func-for-cuda-id)
  (interactive)
  (delete-other-windows)
  (gpu3)
  (comint-delete-output)
  (funcall func-for-cuda-id 1)
  (split-window-right)
  (gpu2)
  (comint-delete-output)
  (funcall func-for-cuda-id 1)
  (split-window-right)
  (gpu1)
  (comint-delete-output)
  (funcall func-for-cuda-id 1)
  (split-window-below)
  (other-window 1)
  (gpu4)
  (comint-delete-output)
  (funcall func-for-cuda-id 2)
  (other-window 1)
  (split-window-below)
  (other-window 1)
  (gpu5)
  (comint-delete-output)
  (funcall func-for-cuda-id 2)
  (other-window 1)
  (split-window-below)
  (other-window 1)
  (gpu6)
  (comint-delete-output)
  (funcall func-for-cuda-id 2)
  (other-window 2)
  (split-window-below)
  (other-window 1)
  (gpu7)
  (comint-delete-output)
  (funcall func-for-cuda-id 3)
  (other-window 2)
  (split-window-below)
  (other-window 1)
  (gpu8)
  (comint-delete-output)
  (funcall func-for-cuda-id 3)
  (other-window 2)
  (split-window-below)
  (other-window 1)
  (gpu9)
  (comint-delete-output)
  (funcall func-for-cuda-id 3)
  (balance-windows)
  )

(defun gpu-farm ()
  (interactive)
  (gpu-farm-int 'identity))
 
(defun gpu-farm-kill ()
  (interactive)
  (gpu-farm-int (lambda (device-id)
		  (setq kill-buffer-query-functions (delq 'process-kill-buffer-query-function kill-buffer-query-functions))
		  (kill-this-buffer)
		  (add-to-list 'kill-buffer-query-functions 'process-kill-buffer-query-function))))

(defun gpu-farm-initialize ()
  (interactive)
  (gpu-farm-int (lambda (device-id)
		  (insert
		   (format "conda deactivate
conda activate ml
cd ~/code/one_off_code/tweet_sentiment_extraction/
for i in $(seq 1 1000) ; do python3 main.py -cuda-device-id %d -hyperparameter-search-roberta ; done" device-id))
		  (comint-send-input))))

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
		    
		    shell))

;; Shortcut Keys

(global-set-key (kbd "C-c \"") 'escape-quotes)
(global-set-key (kbd "C-x C-b") 'buffer-menu)
(global-set-key (kbd "C-x a") 'undo-tree-visualize)
(global-set-key (kbd "C-c `") 'camel-case-to-dashes)
(global-set-key (kbd "C-c ;") 'comment-region)
(global-set-key (kbd "C-c :") 'uncomment-region)
(global-set-key (kbd "<M-down>") 'forward-paragraph)
(global-set-key (kbd "<M-up>") 'backward-paragraph)
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
      (dolist (region-line region-lines)
	(let* ((index-of-first-non-white-space-character (string-match "[^\s-]" region-line))
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

;; Misc. OS-Specific / Machine-Specific Changes

(cond
 ((eq system-type 'darwin)
  (setq mac-command-modifier 'meta)
  (setq mac-function-modifier 'control)
  (setq mac-option-modifier nil)
  (start-cuda-shells))
 ((eq system-type 'gnu/linux)
  )
 (t
  (format "Could not determine OS flavor.")))

;; Continuous Functions

(setq hourly-timer (run-at-time nil 3600 (lambda ()
					   (let ((default-directory "~/"))
					     (async-shell-command "(cd ~/code/one_off_code/ ; git pull)"))
					   )))

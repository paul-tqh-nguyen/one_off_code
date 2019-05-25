;; Standard Settings

(setq inhibit-startup-screen t)
(load-theme 'manoj-dark)
(show-paren-mode 1)

;; Undo Tree

(add-to-list 'load-path "/home/paulnguyen/code/")
(load "undo-tree.el")
(global-undo-tree-mode)

;; Javascript Settings

(load "rjsx-mode.el")
(add-to-list 'auto-mode-alist '("\\.js\\'" . rjsx-mode))

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

(defun python ()
  "Create a new shell buffer with the name \"*python*\" if it doesn't exist. If it does and is running, go it. If it exists and is not running, start it."
  (interactive)
  (start-shell-buffer-with-name "*python*"))

(defun server ()
  "Create a new shell buffer with the name \"*server*\" if it doesn't exist. If it does and is running, go it. If it exists and is not running, start it."
  (interactive)
  (start-shell-buffer-with-name "*server*"))

;; Shortcut Keys

(global-set-key (kbd "C-c \"") 'escape-quotes)
(global-set-key (kbd "C-x C-b") 'buffer-menu)
(global-set-key (kbd "C-x a") 'undo-tree-visualize)
;;(global-set-key (kbd "C-c b") 'switch-to-python-buffer) ;; @todo get this working
(put 'downcase-region 'disabled nil)
(put 'upcase-region 'disabled nil)

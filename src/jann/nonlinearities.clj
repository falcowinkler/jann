(ns jann.nonlinearities
  (:use [clojure.core.matrix]))

(defn- sig [x] (/ 1 (+ 1 (Math/exp (- x)))))
(defn- sig_deriv [x] (* (sig x) (- 1 (sig x))))

(defn sigmoid [x]
  (emap sig x))

(defn sigmoid-derivative [x]
  (emap sig_deriv x))
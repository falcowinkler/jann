(ns jann.initialization
  (:use [clojure.core.matrix]
        [clojure.core.matrix.random]))


(defn random-init [zero]
  (sub (mul 2 (sample-uniform (shape zero))) 1))
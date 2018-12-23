(ns jann.common
  (:use [clojure.core.matrix]))

(defn wx+b [w x b]
  (add (mmul w x) b))

(defn reshape-1 [x]
  "reshapes a tensor of shape [x y ... z] into [x y ... z 1]"
  (reshape x (conj (shape x) 1)))
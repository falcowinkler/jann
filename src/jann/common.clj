(ns jann.common
  (:use [clojure.core.matrix]))

(defn wx+b [w x b]
  (add (mmul w x) b))

(defn reshape-1 [x]
  (reshape x (conj (shape x) 1)))
(ns jann.error-functions
  (:use [clojure.core.matrix]))

(defn mse-derivative [actual expected]
  (sub actual expected))
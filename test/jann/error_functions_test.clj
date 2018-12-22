(ns jann.error-functions-test
  (:require [clojure.test :refer :all]
            [jann.error-functions :as e]))
(use 'clojure.core.matrix)

(deftest test-mse-derivative
  (testing "if mse derivative works"
    (is (= (array [0 -1 0.5])
           (e/mse-derivative (array [5 6 1.5]) (array [5 7 1]))))))
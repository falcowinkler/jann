(ns jann.backpropagation-test
  (:require
    [clojure.test :refer :all]
    [jann.backpropagation :as backprop]
    [jann.nonlinearities :as nonlin]
    [jann.network :as net])
  (:use [clojure.core.matrix]))

; These tests use either per-hand calculated values or
; expected values from a reference implementation

(def test-network (net/make-network [2 3 1] (fn [x] (add x 1))))

(deftest test-calc-logits
  (testing "if calculating logits works"
    (is (= [(array [15.0 17.0])]
           (backprop/calc-logits
             [(array [[1 4] [5 2]])]
             [[1 1]]
             [2 3])))))

(deftest test-calc-activations
  (testing "if calculating activations works"
    (is (= (nonlin/sigmoid [(array [15 17])])
           (backprop/calc-activations
             (backprop/calc-logits
               [(array [[1 4] [5 2]])]
               [[1 1]]
               (vec [2 3])))))))

(deftest test-output-error
  (testing "if saturated neuron has zero gradient"
    (is (= (array [0.0 0.0])
           (backprop/output-error
             [(array [1 1])]
             [(array [5000 5000])]
             (array [0 0])))))
  (testing "if base case works"
    (is (= (array [0.125 0.125])
           (backprop/output-error
             [(array [0.5 0.5])]
             [(array [0 0])]
             (array 0 0))))))

(deftest test-layer-error
  (testing "if error in hidden layer is calculated correctly"
    (is (= [(array [0.03125
                    0.03125])]
           (backprop/layer-errors
             [(array [[0.5 0.5] [0.5 0.5]])]
             (array [0.125 0.125])
             [(array [0 0]) (array [0 0]) (array [0 0])])))))
;
(deftest test-weight-deltas
  (testing "If summing up the deltas for gradient step works"
    (is (= [[-1.0
             -1.0
             [-1.0
              -1.0]
             [[0.0
               0.0]
              [0.0
               0.0]]
             [[1.5
               1.5]
              [1.5
               1.5]] (backprop/nabla-w [[3 3] [2 2] [1 1]]
                                       [[0 0] [-0.5 -0.5] [1.5 1.5] [3 3]])]]))))

;;For the following tests we assume a network that learns XOR
;; 2 inputs, 3 hidden and 1 output)
(deftest get-gradient-test
  (testing "if the backpropagation work. Values from correct reference implementation"
    (let [result (backprop/get-gradient
                   (:weights test-network)
                   (:biases test-network)
                   (array [1 0])
                   [0])]
      (is (= [3 2] (shape (first (:nabla_w result)))))
      (is (= [1 3] (shape (second (:nabla_w result)))))
      (is (= (array [[0.021346298065922878
                      0.021346298065922878
                      0.021346298065922878]]) (array (second (:nabla_w result)))))
      (is (= (array [[0.005020473763405151
                      0.0]
                     [0.005020473763405151
                      0.0]
                     [0.005020473763405151
                      0.0]]) (array (first (:nabla_w result))))))))



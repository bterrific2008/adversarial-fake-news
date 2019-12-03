import pickle
import timeit
from random import shuffle, choice

import tensorflow as tf
import numpy as np
from keras.preprocessing import sequence
from tensorflow.python.ops.parallel_for.gradients import jacobian

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

if __name__ == '__main__':

    N = 500  # Input size
    H = 100  # Hidden layer size
    M = 10  # Output size
    w1 = np.random.randn(N, H)  # first affine layer weights
    b1 = np.random.randn(H)  # first affine layer bias
    w2 = np.random.randn(H, M)  # second affine layer weights
    b2 = np.random.randn(M)  # second affine layer bias

    from tensorflow.keras.layers import Dense

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    model = tf.keras.Sequential()
    model.add(Dense(H, activation='relu', use_bias=True, input_dim=N))
    model.add(Dense(M, activation='softmax', use_bias=True, input_dim=H))
    model.get_layer(index=0).set_weights([w1, b1])
    model.get_layer(index=1).set_weights([w2, b2])


    def jacobian_tensorflow(x):
        jacobian_matrix = []
        for m in range(M):
            # We iterate over the M elements of the output vector
            grad_func = tf.gradients(model.output[:, m], model.input)
            gradients = sess.run(grad_func, feed_dict={model.input: x.reshape((1, x.size))})
            jacobian_matrix.append(gradients[0][0, :])

        return np.array(jacobian_matrix)


    def is_jacobian_correct(jacobian_fn, ffpass_fn):
        """ Check of the Jacobian using numerical differentiation
        """
        x = np.random.random((N,))
        epsilon = 1e-5

        """ Check a few columns at random
        """
        for idx in np.random.choice(N, 5, replace=False):
            x2 = x.copy()
            x2[idx] += epsilon
            num_jacobian = (ffpass_fn(x2) - ffpass_fn(x)) / epsilon
            computed_jacobian = jacobian_fn(x)

            if not all(abs(computed_jacobian[:, idx] - num_jacobian) < 1e-3):
                return False
        return True


    def ffpass_tf(x):
        """ The feedforward function of our neural net
        """
        xr = x.reshape((1, x.size))
        return model.predict(xr)[0]


    is_jacobian_correct(jacobian_tensorflow, ffpass_tf)

    print("Load Model")
    model = pickle.load(open("lstm_model.pickle", "rb"))
    word_bank = pickle.load(open("word_bank.pickle", "rb"))
    encode_word_bank = dict((v, k) for k, v in word_bank.items())

    sentence = """Coulter Calls Out ’Beta Males’ for Threatening ’Rodney King Riots’ Over 
    Berkeley Speech - Breitbart,Jeff Poor,"Thursday on Fox News Channel’s “Hannity,” conservative 
    commentator Ann Coulter, author of “In Trump We Trust: E Pluribus Awesome!” took aim at what 
    she described as “beta males” behind what could be “Rodney King riots” staged to protest a 
    speech she has pledged to give next week at the University of California in Berkeley, 
    CA.  After having her speech canceled by   officials due to safety concerns and then only to 
    have those same officials propose an alternate date for her speech, Coulter dismissed the 
    claim security as a reason for any cancellation. She labeled the protesters as “beta males” 
    with weapons behind what could be “Rodney King riots” allowed by the local police department. 
    “[N]one of this has to do with security,” she said. “After acceding to all their 
    requirements, which were also arbitrary and silly, and they claimed it was on the basis of 
    safety, I suggested two measures that actually would allow free speech to exist on Berkeley 
    if they wanted it to. And that was one thing to announce that any students caught engaging in 
    violence, mayhem, or disrupting an invited speaker’s speech would be expelled. And number 
    two, to have a little talk with the Berkeley chief of police, who is allowing these Rodney 
    King riots to go on whenever conservative speaker speaks. ” “I mean, it is anarchy when you 
    are only enforcing the law in order to allow liberals to speak,” she continued. “But, no, 
    we’ll let these masked rioters show up with weapons and start  —   I mean, they are all 
    little beta males, but with a weapon, even a beta male can do some damage, especially to a   
    girl. To have them stepping in, those private individuals, according to courts, are acting 
    under color of state law. And for the police to refuse to protect even offensive speech and 
    by the way, and I’m the author of 12 New York Times  . This has damaged my reputation for 
    them to be acting like I’m David Duke out there. But courts have found, even somebody out 
    burning an American flag, the police cannot stand by and let skinheads beat them up. That is 
    viewpoint discrimination. And they are all liable. ” (  RCP Video) Follow Jeff Poor on 
    Twitter @jeff_poor """
    tokens = sentence.lower().split()
    # Encode the sentences
    i = 0
    while i < len(tokens):
        if tokens[i] in word_bank:
            tokens[i] = word_bank[tokens[i]]
            i += 1
        else:
            del tokens[i]

    # Truncate and pad input sequences
    print("Truncate and pad input sequences")
    max_review_length = 500
    embedding_vecor_length = 32
    X_value = sequence.pad_sequences([tokens], maxlen=max_review_length)

    X_tensor = tf.convert_to_tensor(X_value)
    with tf.GradientTape() as gtape:
        gtape.watch(X_tensor)
        embedding_output = model.layers[0](X_tensor)
        lstm_output = model.layers[1](embedding_output)
        dense_output = model.layers[2](lstm_output)

    jacobian_matrix = jacobian(dense_output, embedding_output)[0][0].eval()

    sess = tf.InteractiveSession()
    grad_func = tf.gradients(model.output, model.trainable_weights)
    sess.run(tf.initialize_all_variables())
    gradients = sess.run(grad_func, feed_dict={model.input: X_value.reshape((1, X_value.size))})

    embeddings = model.layers[0].get_weights()[0]
    words_embeddings = {w: embeddings[idx] for w, idx in word_bank.items()}
    encode_embeddings = {i: embeddings[i] for i in range(len(embeddings))}

    start = timeit.timeit()
    print("Looking for Adversarial Example")

    true_label = model.predict_classes(X_value)[0][0]
    generated_x = X_value.copy()
    print(model.predict_proba(generated_x))
    update_words = []
    while model.predict_classes(generated_x)[0][0] == true_label:
        # obtain all tokens in our input that aren't blank
        word_indicies = np.nonzero(generated_x[0])[0]

        # add a new word to the front of our tokens if it doesn't already exist
        """if len(word_indicies) < 500:
            word_indicies = np.append(word_indicies, word_indicies[0] - 1)"""
        word_idx = choice(list(word_indicies))

        weight_shift = []
        for word in word_bank.values():
            weight_shift.append((
                np.linalg.norm(
                    np.sign(encode_embeddings[generated_x[0][word_idx]]
                            - encode_embeddings[word])
                    - np.sign(jacobian_matrix[0][word_idx] * (1 if true_label == 1 else -1))),
                word
            ))
        _, update_word = min(weight_shift)
        generated_x[0][word_idx] = update_word
        update_words.append(update_word)

        """words = np.ma.masked_equal(generated_x, 0)
        words = words.compressed()
        print([encode_word_bank[word] for word in words])"""
        print(model.predict_proba(generated_x))

    end = timeit.timeit()
    print(end - start)

    words = np.ma.masked_equal(generated_x, 0)
    words = words.compressed()
    print([encode_word_bank[word] for word in words])
    print([encode_word_bank[word] for word in update_words])

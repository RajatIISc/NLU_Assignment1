import tensorflow as tf
import numpy as np
from nltk.corpus import reuters
import keras.preprocessing.text
import math



train_data = []
with open("questions-words.txt") as data:
    for count,line in enumerate(data,1):
        x=line.split(' ')
        if x[0]!=':':
            train_data.append(x)
        
pair_of_words = list()
for i in range(len(train_data)):    
    for item in list(zip(train_data[i][0:-1],train_data[i][1:])):
        pair_of_words.append(item )
    for item in list(zip(train_data[i][0:-2],train_data[i][2:])):
        pair_of_words.append(item )
    k = list(ele for ele in reversed(train_data[i]))
    for item in list(zip(k[0:-1],k[1:])):
        pair_of_words.append(item )
    for item in list(zip(k[0:-2],k[2:])):
        pair_of_words.append(item )
        
#creating list of words                                                                     
words = set()
for i in range(len(train_data)):
    k = set(train_data[i])
    words.update(k)
words  = list(words)

#calculating vocab_size
vocab_size = len(words)

#create dictionary of vocabulary
word2int = {}
int2word = {}
for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word
    
#creating y_train and x_train    
y_train = list()
for i in range(len(pair_of_words)):
    y_train.append(pair_of_words[i][1])
x_train = list()
for i in range(len(pair_of_words)):
    x_train.append(pair_of_words[i][0])    

#converting them to integers
inputs = list()
labels = list()
for i in range(len(x_train)):
    inputs.append(word2int[x_train[i]])
    labels.append(word2int[y_train[i]])
inputs = np.array(inputs)

def euclidean_dist(a,b):
    return np.linalg.norm(a-b)

def find_closest(query_vector, vectors,client,host,ques):
    min_dist = 10000 # to act like positive infinity
    min_index = -1

    #query_vector = vectors[word_index]

    for index, vector in enumerate(vectors):
        if index!=word2int[ques] and index!=word2int[host]:# and index!=word2int[client]:

            if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
    
                min_dist = euclidean_dist(vector, query_vector)
                min_index = index
            
    return int2word[min_index]

def sim(client,host,ques):
    client1  = E[word2int[client]]
    host1 = E[word2int[host]]
    ques1 = E[word2int[ques]]
    temp = np.add(np.subtract(host1,client1),ques1)
    return find_closest(temp,E,client,host,ques)
#tensorflow stuff
epochs = 50
for batch_size in [32]:
    for num_sampled in [2]:
        for embedding_size in [256]:
            embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocab_size]))
            train_inputs = tf.placeholder(tf.int32, shape=[None,])
            train_labels = tf.placeholder(tf.int32, shape=[None,1])
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init) #make sure you do this!
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=train_labels,inputs=embed,num_sampled=num_sampled,num_classes=vocab_size))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(loss)
            n_iters = int(np.ceil(len(inputs)/batch_size))
            print(n_iters)
            # train for n_iter iterations
            labels = np.array(labels)
            for cnt in range(epochs):
                index= 0
                for i in range(n_iters):
                    sess.run(optimizer, feed_dict={train_inputs: inputs[index:index+batch_size], train_labels: np.expand_dims(labels[index:index+batch_size],axis=1)})
                    index = index + batch_size
                print('loss is : ', sess.run(loss, feed_dict={train_inputs: inputs[0:batch_size], train_labels: np.expand_dims(labels[0:batch_size],axis=1)}))
            
                print('epoch %d done'%cnt)    
            #vectors = np.zeros((vocab_size,embedding_size))   
            #B = np.array(sess.run(nce_biases))
            #W = np.array(sess.run(nce_weights))
            #for i in range(embedding_size):
            #    vectors[:,i] = W[:,i]+B
            
            
            E = np.array(sess.run(embeddings))
            text = E.tolist()
            
            for i,item in enumerate(text):
                item.insert(0,int2word[i] )
            
            EF = [' '.join(map(str,item)) for item in text]
            myfile = open("task2.txt","w")
            
            for item in EF:
                myfile.write("%s\n" % item)
#def cosine_sim(client,host,ques):
#    client  = np.array(E[word2int[client]])
#    host = np.array(E[word2int[host]])
#    ques = np.array(E[word2int[ques]])
#    train_vector = np.subtract(client,host)
#    max_dis = 0
#    for i,item in enumerate(E) :
#        ans = np.array(item)
#        test_vector = np.subtract(ques,ans)
#        cos_dis = np.dot(train_vector, test_vector)/(np.linalg.norm(train_vector)* np.linalg.norm(test_vector))
#        if cos_dis>max_dis:
#            max_dis=cos_dis
#            max_index = i
#    return int2word[i]
            
    
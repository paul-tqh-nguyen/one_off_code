// List Library

/*
TODO:
    add insert functions to all     
    make sure all prevs and next update correctly for doubly linked list
    make pops return stuff
    remove pops from the pop tails and such
    add private stuff to reduce code liek to check if we're out of space etc. 
    make increase space private
    make sure all tail and head stuff is consistent and checked on every method
    recheck and add more assrts to [] operators and insert function
    
    Write a PNG parser
*/

#ifndef _LIST_H_
#define _LIST_H_

#ifndef TEST
#define TEST(x) cout << (#x) << ": " << x << endl
#endif

#ifndef MOD
#define MOD(a,b) ( ((((int)a)%((int)b))+((int)b))%((int)b) )
#endif

#include <iostream>
#include <stdio.h>
#include <assert.h>

#define DEBUG BUILD_DEBUG
#define ASSERT(x, msg) if (DEBUG && !(x)) { fprintf(stderr, "Assertion failed in %s(%d): %s\n", __FILE__, __LINE__, msg); assert(false); exit(1); }

using std::cout;
using std::endl;

template<class real> 
class List { 
    
    public:
        unsigned int length;
        virtual void push(real new_value) = 0; // pushes to the front (position 0)
        virtual void pop() = 0; // pops from the front (position 0)
        virtual void append(real new_value) = 0; // tacks onto the end (position length-1)
        virtual void pop_tail() = 0; // pops from the end (position length-1)
        virtual void insert(unsigned int index) = 0; // inserts element at index
        virtual void remove(unsigned int index) = 0; // removes element at index
        virtual void clear() = 0; // removes all elements
        virtual real& operator[] (const int index) = 0; // subscripting operator
        virtual void print(){ // prints elements of the list in square brackets delimited by commas
            // NOTE: Slow and possibly quadratic time implementation if the implementation of operator[]() is takes linear time
            // Recommended to override (or not since printing is usually for debugging, where speed isn't a concern)
            cout << "[";
            for(int i = 0; i < this->length; i++){ 
                cout << (*this)[i];
                if (i != length-1){
                    cout << ", ";
                }
            }
            cout << "]" << endl;
        }
};

template<class real> 
class LinkedList : public List<real> {
    
    class Node{ 
        public:
            real value;
            Node* next;
            
            Node(real init_value, Node* next_node=NULL){
                value = init_value;
                next = next_node;
            }
    };
    
    public:
        Node* head;
        Node* tail;
        using List<real>::length;
        
        LinkedList(){
            length = 0;
            head = NULL;
            tail = NULL;
        }
        
        void push(real new_value){
            Node* new_node = new Node(new_value, head);
            head = new_node;
            length++;
            if (length == 1) {
                tail = head;
            }
            ASSERT(head!=NULL, "NULL head after pushing an element");
            ASSERT(tail!=NULL, "NULL tail after pushing an element");
        }
        
        void pop(){
            ASSERT(length > 0, "Cannot pop from empty list");
            Node* next_element = head->next;
            delete head;
            head = next_element;
            length--;
            ASSERT(length >= 0, "Cannot have negative length");
            if (length == 0) {
                ASSERT(head==NULL, "List length of zero, but non-NULL head");
                tail = NULL;
            }
        }
        
        void append(real new_value){ 
            Node* new_node = new Node(new_value, NULL);
            if (length > 0) {
                ASSERT(tail->next == NULL, "Non-NULL tail->next");
                tail->next = new_node;
            }
            tail = new_node;
            length++;
            if (length == 1) {
                ASSERT(head == NULL, "Non-NULL head with length zero")
                head = tail;
            }
        }
        
        void pop_tail(){ // pops from the end ::: NOTE: Linear Time
            ASSERT(length > 0, "Cannot pop the tail off of empty list");
            if (length == 1){
                this->pop();
                return;
            }
            Node* second_to_last_element = head;
            for(int i = 1; i < length-1; i++){
                second_to_last_element = second_to_last_element->next;
                ASSERT(second_to_last_element != NULL, "Non-NULL next pointer on element that is not the tail");
            }
            delete tail;
            tail = second_to_last_element;
            tail->next = NULL;
            length--;
            ASSERT(length >= 0, "Cannot have negative length");
        }
        
        real& operator[] (const int index) { // NOTE: Linear Time
            ASSERT(index < length, "Index is out of bounds for operator[](). The index value is greater than or equal to the list length");
            ASSERT(index >= 0, "Index is out of bounds for operator[](). The index value is less than zero");
            Node* needed_element = head;
            for(int i = 0; i < index; i++){
                ASSERT(needed_element != NULL, "NULL next pointer for non-tail element");
                needed_element = needed_element->next;
            }
            return needed_element->value;
        }
        
        void remove(unsigned int index){ // NOTE: Linear Time
            ASSERT(index < length, "Index is out of bounds for remove(). The index value is greater than or equal to the list length");
            ASSERT(index >= 0, "Index is out of bounds for remove(). The index value is less than zero");
            if (length == 1 || index == 0){ // to void having to change head pointer here in this code
                this->pop();
                return;
            }
            if (index == length-1) { // to void having to change tail pointer here in this code
                this->pop_tail();
                return;
            }
            Node* element_before_element_to_be_removed = head;
            Node* element_to_be_removed = head->next;
            for(int i = 1; i < index; i++){
                ASSERT(element_to_be_removed != NULL, "NULL next pointer for non-tail element");
                ASSERT(element_before_element_to_be_removed != NULL, "NULL next pointer for non-tail element");
                element_before_element_to_be_removed = element_to_be_removed;
                element_to_be_removed = element_to_be_removed->next;
            }
            element_before_element_to_be_removed->next = element_to_be_removed->next;
            delete element_to_be_removed;
            length--;
            ASSERT(length >= 0, "Cannot have negative length");
        }
        
        void clear(){
            while (head != NULL){
                this->pop();
            }
            ASSERT(length==0, "Number of list elements cleared does not match list length");
        }
        
        ~LinkedList(){
            this->clear();
            ASSERT(length==0, "Number of list elements deallocated does not match list length");
        }
};

template<class real> 
class DoublyLinkedList : public List<real> {
    
    class Node{
        public:
            real value;
            Node* next;
            Node* prev;
            
            Node(real init_value, Node* prev_node=NULL, Node* next_node=NULL){
                value = init_value;
                next = next_node;
                prev = prev_node;
            }
    };
    
    public:
        Node* head;
        Node* tail;
        using List<real>::length;
        
        DoublyLinkedList(){
            length = 0;
            head = NULL;
            tail = NULL;
        }
        
        void push(real new_value){
            Node* new_node = new Node(new_value, NULL, head);
            if (head != NULL){
                head->prev = new_node;
            }
            head = new_node;
            length++;
            if (length == 1) {
                tail = head;
            }
            ASSERT(head!=NULL, "NULL head after pushing an element");
            ASSERT(tail!=NULL, "NULL tail after pushing an element");
        }
        
        void pop(){
            ASSERT(length > 0, "Cannot pop from empty list");
            ASSERT(head->prev==NULL, "Non-NULL head->prev"); 
            Node* next_element = head->next;
            delete head;
            head = next_element;
            length--;
            ASSERT(length >= 0, "Cannot have negative length");
            if (length == 0) {
                ASSERT(head==NULL, "List length of zero, but non-NULL head");
                tail = NULL;
            } else {
                ASSERT(head!=NULL, "NULL head in non-empty list");
                head->prev = NULL;
            }
        }
        
        void append(real new_value){ 
            Node* new_node = new Node(new_value, tail, NULL);
            if (length > 0) {
                ASSERT(tail->next == NULL,"Non-NULL tail->next")
                tail->next = new_node;
            }
            tail = new_node;
            length++;
            if (length == 1) {
                ASSERT(head == NULL, "Non-NULL head with length zero")
                head = tail;
            }
            ASSERT(head->prev == NULL, "Non-NULL head->prev")
            ASSERT(tail->next == NULL, "Non-NULL tail->next")
        }
        
        void pop_tail(){ 
            ASSERT(length > 0, "Cannot pop the tail off of empty list");
            if (length == 1){
                this->pop();
                return;
            }
            Node* second_to_last_element = tail->prev;
            delete tail;
            tail = second_to_last_element;
            tail->next = NULL;
            length--;
            ASSERT(length >= 0, "Cannot have negative length");
        }
        
        real& operator[] (const int index) { // NOTE: Linear Time
            ASSERT(index < length, "Index is out of bounds for operator[](). The index value is greater than or equal to the list length");
            ASSERT(index >= 0, "Index is out of bounds for operator[](). The index value is less than zero");
            Node* needed_element = head;
            for(int i = 0; i < index; i++){
                ASSERT(needed_element != NULL, "NULL next pointer for non-tail element or NULL head in non-empty list");
                needed_element = needed_element->next;
                ASSERT(needed_element->prev != NULL, "NULL prev pointer for non-head element");
            }
            return needed_element->value;
        }
        
        void remove(unsigned int index){ // NOTE: Linear Time
            ASSERT(index < length, "Index is out of bounds for remove(). The index value is greater than or equal to the list length");
            ASSERT(index >= 0, "Index is out of bounds for remove(). The index value is less than zero");
            if (length == 1 || index == 0){ // to void having to change head pointer here in this code
                this->pop();
                return;
            }
            if (index == length-1) { // to void having to change tail pointer here in this code
                this->pop_tail();
                return;
            }
            Node* element_to_be_removed = head->next;
            for(int i = 1; i < index; i++){
                ASSERT(element_to_be_removed != NULL, "NULL next pointer for non-tail element");
                ASSERT(element_to_be_removed->prev != NULL, "NULL prev pointer for non-head element");
                element_to_be_removed = element_to_be_removed->next;
            }
            element_to_be_removed->prev->next = element_to_be_removed->next;
            element_to_be_removed->next->prev = element_to_be_removed->prev;
            delete element_to_be_removed;
            length--;
            ASSERT(length >= 0, "Cannot have negative length");
        }
        
        void clear(){
            while (head != NULL){
                this->pop();
            }
            ASSERT(length==0, "Number of list elements cleared does not match list length");
        }
        
        ~DoublyLinkedList(){
            this->clear();
            ASSERT(length==0, "Number of list elements deallocated does not match list length");
        }
};

template<class real> 
class ArrayList : public List<real> {
    
    public:
        int head;
        int tail;
        using List<real>::length;
        real *data;
        unsigned int available_space;
        
        ArrayList(){
            length = 0;
            available_space = 1;
            data = new real[available_space];
            head = -1;
            tail = -1;
        }
        
        ~ArrayList(){
            delete[] data;
        }
        
        void clear(){
            length = 0;
            head = -1;
            tail = -1;
        }
        
        void increase_space(){ // NOTE: Takes linear time
            ASSERT(length <= available_space, "Recorded length is larger than the amount of space available");
            real* new_data = new real[this->available_space*2];
            unsigned int count = 0;
            for(int i = head; MOD(tail-i,this->available_space) != this->available_space-1 || count==0; i = MOD(i+1,this->available_space)) {
                new_data[count] = data[i];
                count++;
            }
            ASSERT(count == length, "Number of elements counted inconsistent with length");
            delete[] data;
            data = new_data;
            head = 0;
            tail = count-1;
            this->available_space *= 2;
        }
        
        void decrease_space(){ // NOTE: Takes linear time
            real* new_data = new real[this->length];
            unsigned int count = 0;
            for(int i = head; MOD(tail-i,available_space) != available_space-1 || count==0; i = MOD(i+1,available_space)) {
                new_data[count] = data[i];
                count++;
            }
            ASSERT(count == length, "Number of elements counted inconsistent with length");
            delete[] data;
            data = new_data;
            head = 0;
            tail = count-1;
            available_space = length;
        }
        
        void push(real new_value){
            if (length == 0){
                ASSERT(head == -1, "Zero length with head at a valid index");
                ASSERT(tail == -1, "Zero length with tail at a valid index");
                head = 0;
                tail = 0;
                data[0] = new_value;
                length++;
                return;
            }
            if( MOD(tail-head,available_space)==available_space-1 ){
                this->increase_space();
            }
            head = MOD(head-1,available_space);
            data[head] = new_value;
            length++;
            ASSERT(MOD(tail-head,available_space)!=length, "Distance between head and tail inconsistent with length");
        }
        
        void pop(){
            ASSERT(length > 0, "Cannot pop from empty list");
            if (tail == head) { // i.e. one element
                ASSERT(length == 1, "head == tail with legnth > 1");
                this->clear();
                return;
            }
            head = MOD(head+1,available_space);
            length--;
            ASSERT(length >= 0, "Cannot have negative length");
            ASSERT((head != tail && length > 1) || (head == tail && length == 1), "For length == 1, head and tail must be equal. Otherwise, head and tail cannot point to same index with length bigger than 1");
            ASSERT(MOD(tail-head,available_space)!=length, "Distance between head and tail inconsistent with length");
        }
        
        void append(real new_value){
            if (length == 0){
                ASSERT(head == -1, "Zero length with head at a valid index");
                ASSERT(tail == -1, "Zero length with tail at a valid index");
                head = 0;
                tail = 0;
                data[0] = new_value;
                length++;
                return;
            }
            if( MOD(tail-head,available_space)==available_space-1 ){
                this->increase_space();
            }
            tail = MOD(tail+1,available_space);
            data[tail] = new_value;
            length++;
            ASSERT(MOD(tail-head,available_space)!=length, "Distance between head and tail inconsistent with length");
        }
        
        void pop_tail(){
            ASSERT(length > 0, "Cannot pop from empty list");
            if (tail == head) { // i.e. one element
                ASSERT(length == 1, "head == tail with legnth > 1");
                this->clear();
                return;
            }
            tail = MOD(tail-1,available_space);
            length--;
            ASSERT(length >= 0, "Cannot have negative length");
            ASSERT((head != tail && length > 1) || (head == tail && length == 1), "For length == 1, head and tail must be equal. Otherwise, head and tail cannot point to same index with length bigger than 1");
            ASSERT(MOD(tail-head,available_space)!=length, "Distance between head and tail inconsistent with length");
        }
        
        real& operator[] (const int index) { // NOTE: Linear Time
            ASSERT(index < length, "Index is out of bounds for operator[](). The index value is greater than or equal to the list length");
            ASSERT(index >= 0, "Index is out of bounds for operator[](). The index value is less than zero");
            return data[MOD(head+index,available_space)];
        }
        
        void remove(unsigned int index){ 
            ASSERT(index < length, "Index is out of bounds for remove(). The index value is greater than or equal to the list length");
            ASSERT(index >= 0, "Index is out of bounds for remove(). The index value is less than zero");
            if (length == 1){ 
                ASSERT(head == tail, "head != tail with legnth == 1");
                this->pop();
                return;
            }
            for(int i = MOD(head+index,available_space); MOD(tail-i,available_space) != available_space-1; i = MOD(i+1,available_space)) {
                data[i] = data[MOD(i+1,available_space)];
            }
            tail = MOD(tail-1,available_space);
            length--;
            ASSERT(length >= 0, "Cannot have negative list length");
            ASSERT(MOD(tail-head,available_space)!=length, "Distance between head and tail inconsistent with length");
        }
};

void list_tester(){
    //auto list = LinkedList<double>();
    //auto list = DoublyLinkedList<double>();
    auto list = ArrayList<double>();
    cout << endl;
    
    // Test append()
    cout << "Testing append()" << endl;
    for(int i = 0; i < 5; i++){
        list.append(i);
    }
    cout << "Expected: [0, 1, 2, 3, 4]" << endl;
    cout << "Result:   ";
    list.print();
    cout << endl;
    
    // Test pop_tail()
    cout << "Testing pop_tail()" << endl; 
    cout << "Expected: " << endl;
    cout << "length: 5" << endl;
    cout << "[0, 1, 2, 3, 4]" << endl;
    cout << "length: 4" << endl;
    cout << "[0, 1, 2, 3]" << endl;
    cout << "length: 3" << endl;
    cout << "[0, 1, 2]" << endl;
    cout << "length: 2" << endl;
    cout << "[0, 1]" << endl;
    cout << "length: 1" << endl;
    cout << "[0]" << endl;
    cout << "length: 0" << endl;
    cout << "[]" << endl;
    cout << endl;
    cout << "Result: " << endl;
    for(int i = 0; i < 5; i++){ 
        cout << "length: " << list.length << endl;
        list.print();
        list.pop_tail();
    }
    cout << "length: " << list.length << endl;
    list.print();
    cout << endl;
    
    // Test clear()
    cout << "Testing clear()" << endl;
    for(int i = 0; i < 5; i++){
        list.append(i);
    }
    list.clear();
    cout << "Expected: []" << endl;
    cout << "Result:   ";
    list.print();
    cout << endl;
    
    // Test push()
    cout << "Testing push()" << endl;
    list.clear();
    for(int i = 0; i < 5; i++){
        list.push(i);
    }
    cout << "Expected: [4, 3, 2, 1, 0]" << endl;
    cout << "Result:   ";
    list.print();
    cout << endl;
    
    // Test pop()
    cout << "Testing pop()" << endl; 
    cout << "Expected: " << endl;
    cout << "length: 5" << endl;
    cout << "[4, 3, 2, 1, 0]" << endl;
    cout << "length: 4" << endl;
    cout << "[3, 2, 1, 0]" << endl;
    cout << "length: 3" << endl;
    cout << "[2, 1, 0]" << endl;
    cout << "length: 2" << endl;
    cout << "[1, 0]" << endl;
    cout << "length: 1" << endl;
    cout << "[0]" << endl;
    cout << "length: 0" << endl;
    cout << "[]" << endl;
    cout << endl;
    cout << "Result: " << endl;
    for(int i = 0; i < 5; i++){
        cout << "length: " << list.length << endl;
        list.print();
        list.pop();
    }
    cout << "length: " << list.length << endl;
    list.print();
    cout << endl;
    
    // Test operator[]()
    cout << "Testing operator[]()" << endl; 
    list.clear();
    for(int i = 0; i < 5; i++){
        list.append(i);
    }
    cout << "Expected: 0123402468" << endl;
    cout << "Result:   ";
    for(int i = 0; i < 5; i++){
        cout << list[i];
        list[i] *= 2;
    }
    for(int i = 0; i < 5; i++){
        cout << list[i];
    }
    cout << endl;
    cout << endl;
    
    // Test remove()
    cout << "Testing remove()" << endl;
    cout << "Expected: " << endl;
    cout << "[0, 1, 2, 3, 4]" << endl;
    cout << "[0, 1, 2, 3]" << endl;
    cout << "[1, 2, 3]" << endl;
    cout << "[1, 3]" << endl;
    cout << "[1]" << endl;
    cout << "[]" << endl;
    cout << "Result:   " << endl;
    list.print();
    list.remove(list.length-1);
    list.print();
    list.remove(0); 
    list.print();
    list.remove(1); 
    list.print();
    list.remove(list.length-1); 
    list.print(); 
    list.remove(0); 
    list.print(); 
    cout << endl;
    
    // Test to make sure everything works
    cout << "Test to make sure everything works" << endl;
    list.clear();
    list.push(0);
    for(int i = 0; i < 5; i++){
        list.append(i+1);
        list.push(i+1);
    }
    cout << "Expected: " << endl;
    cout << "[5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5]" << endl;
    cout << "[5, 4, 3, 2, 1, 0, 1, 2, 3, 4]" << endl;
    cout << "[4, 3, 2, 1, 0, 1, 2, 3, 4]" << endl;
    cout << "[4, 3, 2, 1, 0, 1, 2, 3]" << endl;
    cout << "[3, 2, 1, 0, 1, 2, 3]" << endl;
    cout << "[3, 2, 1, 0, 1, 2]" << endl;
    cout << "[2, 1, 0, 1, 2]" << endl;
    cout << "[2, 1, 0, 2]" << endl;
    cout << "[2, 0, 2]" << endl;
    cout << "[4, 0, 4]" << endl;
    cout << "[]" << endl;
    cout << "Result:   " << endl;
    list.print(); 
    list.remove(list.length-1); 
    list.print();
    list.remove(0); 
    list.print();
    list.remove(list.length-1); 
    list.print();
    list.remove(0); 
    list.print();
    list.pop_tail(); 
    list.print(); 
    list.pop(); 
    list.print(); 
    list.remove(3); 
    list.print();
    list.remove(1);
    list.print();
    for(int i = 0; i < list.length; i++){
        list[i] *= 2;
    }
    list.print();
    list.clear();
    list.print();
    cout << endl;
}

#endif // _LIST_H_ 


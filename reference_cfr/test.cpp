#include <pthread.h>
#include <iostream>
#include <mutex>
#include <thread>

using namespace std;

mutex mtx;

void* print_thread_id(void*) {
    thread::id this_id = std::this_thread::get_id();
    mtx.lock();
    cout << "I am " << this_id << endl;
    mtx.unlock();
    return NULL;
}


int main() {
    pthread_t threads[100];
    for (int i = 0; i < 100; i++) {
        pthread_create(&threads[i], NULL, print_thread_id, NULL);
    }

    for (int i = 0; i < 100; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}

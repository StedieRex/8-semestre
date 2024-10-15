
interface  MessageQueue{
    void put(String message);
    String get();
    String pull();
    void notifelisteners();
}

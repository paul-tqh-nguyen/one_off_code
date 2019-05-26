import React from 'react';
import Header from './Components/Header';
import Footer from './Components/Footer';
import ChordEditor from './Components/ChordEditor';

function App() {
    return (
        <div className="App">
          <Header />
	  <h1>Hello World!</h1>
          <div className="chord-editor">
            <ChordEditor />
          </div>
          <Footer />
        </div>
    );
}

export default App;

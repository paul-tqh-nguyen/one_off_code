import React, { Component } from 'react';

export default class Footer extends Component {
    constructor (props) {
        super(props);
        this.state = { year : new Date().getFullYear() };
    }
    
    render() {
        return (
            <footer>
              <h1>
                <ul className="site-link">
                  <li>
                    &copy; {this.state.year} PNguyen
                  </li>
                </ul>
              </h1>
            </footer>
        );
    }
}

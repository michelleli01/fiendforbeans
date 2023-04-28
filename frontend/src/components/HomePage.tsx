import { useState } from 'react';
import logo from './../static/images/logo.png';
import { Bean } from './../types';
import Recommendations from './Recommendations';
import { Link } from 'react-router-dom';

const HomePage = () => {
  const [search, setSearch] = useState<string>('');
  const [roast, setRoast] = useState<string>('Light');
  const [recommend, setRecommend] = useState<Bean[] | undefined>(undefined);

  const updateSearch = (event: { target: { value: string } }) => {
    let query = event.target.value.trim();
    setSearch(query);
  };

  const updateRoast = (event: { target: { value: any } }) => {
    let roast = event.target.value;
    setRoast(roast);
  };

  return (
    <div className='full-body-container'>
      <img className='logo' src={logo} alt='Fiend for Beans logo' />{' '}
      <label>Enter which flavor profiles you enjoy.</label>
      <div className='input-box'>
        <div>
          <input
            placeholder='citrus, floral, sweet'
            id='filter-text-val'
            onChange={updateSearch}
          />
        </div>
      </div>
      <label>Select which roast you prefer.</label>
      <div className='dropdown'>
        <select name='roast' id='roast_select' onChange={updateRoast}>
          <option value='Light'>Light</option>
          <option value='Medium-Light'>Medium-Light</option>
          <option value='Medium'>Medium</option>
          <option value='Medium-Dark'>Medium-Dark</option>
          <option value='Dark'>Dark</option>
        </select>
      </div>
      <br />
      <button id='search-button'>
        <Link
          style={{ textDecoration: 'none', color: 'grey' }}
          to={`/search?q=${search}&roast=${roast}`}
        >
          Search
        </Link>
      </button>
      <div id='answer-box'>
        <Recommendations recommend={recommend} />
      </div>
    </div>
  );
};

export default HomePage;

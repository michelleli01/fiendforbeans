export type Bean = {
  bean_info: {
    id: number;
    name: string;
    roaster: string;
    dollars_per_ounce: number;
    origin: string;
    roaster_link: string;
    roast: string;
    review: string;
  };
  score: number;
};

export type Roast =
  | 'Light'
  | 'Medium-Light'
  | 'Medium'
  | 'Medium-Dark'
  | 'Dark';

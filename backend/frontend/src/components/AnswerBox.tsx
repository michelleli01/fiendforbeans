import React from "react";
import "../styles/AnswerBox.css";
import amavida from "../static/images/roaster_logos/Amavida Coffee Roasters.webp";
import baba from "../static/images/roaster_logos/Baba Java Coffee.webp";
import barrington from "../static/images/roaster_logos/Barrington Coffee Roasting.webp";
import big_island from "../static/images/roaster_logos/Big Island Coffee Roasters.webp";
import bird_rock from "../static/images/roaster_logos/Bird Rock Coffee Roasters.webp";
import cafe_kreyol from "../static/images/roaster_logos/Cafe Kreyol.webp";
import campos from "../static/images/roaster_logos/Campos Coffee.webp";
import corvus from "../static/images/roaster_logos/Corvus Coffee.webp";
import davids_nose from "../static/images/roaster_logos/David's Nose.webp";
import dragonfly from "../static/images/roaster_logos/Dragonfly Coffee Roasters.webp";
import durango from "../static/images/roaster_logos/Durango Coffee Company.webp";
import el_gran from "../static/images/roaster_logos/El Gran Cafe.webp";
import equator from "../static/images/roaster_logos/Equator Coffees.webp";
import equiano from "../static/images/roaster_logos/Equiano Coffee.webp";
import hula from "../static/images/roaster_logos/Hula Daddy Kona Coffee.webp";
import jackrabbit from "../static/images/roaster_logos/Jackrabbit Java.webp";
import jaunt from "../static/images/roaster_logos/Jaunt Coffee Roasters.webp";
import jbc from "../static/images/roaster_logos/JBC.webp";
import klatch from "../static/images/roaster_logos/Klatch Coffee.webp";
import lexington from "../static/images/roaster_logos/Lexington Coffee Roasters.webp";
import magnolia from "../static/images/roaster_logos/Magnolia Coffee.webp";
import merge from "../static/images/roaster_logos/Merge Coffee Company.webp";
import modcup from "../static/images/roaster_logos/modcup.webp";
import monarch from "../static/images/roaster_logos/Monarch Coffee.webp";
import mostra from "../static/images/roaster_logos/Mostra Coffee.webp";
import northbound from "../static/images/roaster_logos/Northbound Coffee Roasters.webp";
import nostalgia from "../static/images/roaster_logos/Nostalgia Coffee Roasters.webp";
import old_world from "../static/images/roaster_logos/Old World Coffee Lab.webp";
import paradise from "../static/images/roaster_logos/Paradise Roasters.webp";
import pts from "../static/images/roaster_logos/PT’s Coffee Roasting Co..webp";
import ramshead from "../static/images/roaster_logos/RamsHead Coffee Roasters.webp";
import red from "../static/images/roaster_logos/Red Rooster Coffee Roaster.webp";
import regent from "../static/images/roaster_logos/Regent Coffee.webp";
import revel from "../static/images/roaster_logos/Revel Coffee.webp";
import rnd from "../static/images/roaster_logos/RND & Red Rooster Coffee Roaster.webp";
import roadmap from "../static/images/roaster_logos/Roadmap CoffeeWorks.webp";
import rusty from "../static/images/roaster_logos/Rusty Dog Coffee.webp";
import san_fran from "../static/images/roaster_logos/San Francisco Bay Coffee Company.webp";
import skytop from "../static/images/roaster_logos/SkyTop Coffee.webp";
import speedwell from "../static/images/roaster_logos/Speedwell Coffee.webp";
import spirit from "../static/images/roaster_logos/Spirit Animal Coffee.webp";
import temple from "../static/images/roaster_logos/Temple Coffee.webp";
import thanksgiving from "../static/images/roaster_logos/Thanksgiving Coffee Company.webp";
import theory from "../static/images/roaster_logos/Theory Coffee Roasters.webp";
import willoughby from "../static/images/roaster_logos/Willoughby's Coffee & Tea.webp";

type AnswerBoxProps = {
  title: string;
  similarity: number;
  roast: string;
  price: number;
  titleDesc: string;
  roasterLink: string;
  roaster: string;
};

const imgs = {
  "Amavida Coffee Roaster": { amavida },
  "Baba Java Coffee": { baba },
  "Barrington Coffee Roasting": { barrington },
  "Big Island Coffee Roasters": { big_island },
};

export default function AnswerBox({
  title,
  similarity,
  roast,
  price,
  roaster,
  titleDesc,
  roasterLink,
}: AnswerBoxProps): JSX.Element {
  console.log(roaster);

  return (
    <div className="box">
      <h2 className="bean-title">{title}</h2>
      <img src={Object.values(roaster)[0]} alt="Coffee Roaster"></img>

      <p className="bean-similarity">
        Percent Confidence: {(similarity * 100).toFixed(2)}%
      </p>
      <p className="bean-similarity">Roast: {roast}</p>
      <p className="bean-similarity">Price per Ounce: ${price}</p>
      <a className="bean-roaster" href={roasterLink}>
        {roaster}
      </a>

      <p className="bean-desc">{titleDesc}</p>
    </div>
  );
}

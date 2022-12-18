import {
  BrowserRouter,
  Routes,
  Route,
} from "react-router-dom";
import ApexChart from "./pages/home/ApexChart";

import Form from "./pages/home/Form";


function App() {
  return (
    <div className="App">
          <BrowserRouter>
    <Routes>

      <Route path="/" >
        <Route index element={<Form/>}></Route>
        <Route path="form" element={<Form/>}></Route>
        <Route path="home" element={<ApexChart/>}/>
      </Route>

    </Routes>
  </BrowserRouter>
    </div>
  );
}

export default App;
